"""Danogo DEX Module.

This module implements support for the Danogo Concentrated Liquidity Market Maker
(CLMM) DEX on the Cardano blockchain.  It provides:

- On-chain ``PlutusData`` types that mirror the Aiken smart-contract datum
  structures, enabling CBOR (de)serialisation via PyCardano.
- Pure math helpers for CLMM swap pricing (virtual-reserve calculation, the
  constant-product formula with LP and platform fees, and exact input search).
- Blockfrost API integration with TTL caching for fetching staking rewards and
  the current chain slot.
- ``ConcentratedPoolState``: the main Pydantic model representing a live pool
  UTXO, with swap-quote (``get_amount_out`` / ``get_amount_in``) and full
  transaction-building (``swap_utxo``) capabilities.
"""

import logging
import math
import os
import time
from dataclasses import dataclass
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from functools import cached_property
from typing import Any
from typing import ClassVar

from blockfrost import ApiError
from blockfrost import BlockFrostApi
from cachetools import TTLCache
from cachetools import cached
from pycardano import Address
from pycardano import Network
from pycardano import PlutusData
from pycardano import Redeemer
from pycardano import ScriptHash
from pycardano import TransactionBuilder
from pycardano import TransactionId
from pycardano import TransactionInput
from pycardano import TransactionOutput
from pycardano import UTxO
from pycardano import VerificationKeyHash
from pycardano.metadata import AuxiliaryData
from pycardano.metadata import Metadata
from pycardano.plutus import PlutusV1Script
from pycardano.plutus import PlutusV2Script
from pycardano.plutus import PlutusV3Script
from pycardano.plutus import plutus_script_hash
from pycardano.transaction import Withdrawals
from pydantic import computed_field

from charli3_dendrite.backend import get_backend
from charli3_dendrite.dataclasses.datums import OrderDatum
from charli3_dendrite.dataclasses.datums import PoolDatum
from charli3_dendrite.dataclasses.models import OrderType
from charli3_dendrite.dataclasses.models import PoolSelector
from charli3_dendrite.dataclasses.models import ScriptReference
from charli3_dendrite.dexs.amm.amm_base import AbstractPoolState
from charli3_dendrite.dexs.core.errors import InvalidPoolError
from charli3_dendrite.utility import Assets
from charli3_dendrite.utility import asset_to_value
from charli3_dendrite.utility import naturalize_assets

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("charli3_dendrite")

# Maximum iterations for the exponential search in get_amount_in.

_MAX_SEARCH_ITERATIONS = 200
AIKEN_TUPLE_LENGTH = 2
Basis = int

# =============================================================================
# 1. ON-CHAIN PLUTUS DATA TYPES
#    Python mirrors of the Aiken on-chain types; used for CBOR (de)serialisation.
# =============================================================================


@dataclass
class DanogoTupleAsset:
    """Python mirror of the Aiken ``TupleAsset`` type: ``(PolicyId, AssetName)``."""

    policy: bytes  # maps to Aiken Index 0 (PolicyId)
    asset_name: bytes  # maps to Aiken Index 1 (AssetName)

    @property
    def unit(self) -> str:
        """Return the Cardano asset unit string.

        Format: ``<policy_id_hex><asset_name_hex>``.
        """
        if not self.policy:
            return "lovelace"
        return self.policy.hex() + self.asset_name.hex()

    @property
    def assets(self) -> Assets:
        """Return a zero-quantity ``Assets`` placeholder for this asset.

        Used when only the asset *identity* (unit string) is needed, e.g.
        when building the asset pair returned by
        ``ConcentratedPoolDatum.pool_pair``.
        """
        return Assets(root={self.unit: 0})

    @classmethod
    def from_list(cls, data: list[bytes]) -> "DanogoTupleAsset":
        """Construct a ``DanogoTupleAsset`` from a raw CBOR list.

        Args:
            data: Exactly two ``bytes`` objects: ``[policy_id, asset_name]``.

        Returns:
            A new ``DanogoTupleAsset`` instance.

        Raises:
            ValueError: If ``data`` does not contain exactly 2 elements.
        """
        if len(data) != AIKEN_TUPLE_LENGTH:
            raise ValueError(
                "Aiken Tuple (PolicyId, AssetName) must have exactly 2 elements.",
            )
        return cls(policy=data[0], asset_name=data[1])


@dataclass
class DanogoPRational(PlutusData):
    """Python representation of Aiken PRational type.

    Maps to: pub type PRational = { numerator: Int, denominator: Int }
    """

    CONSTR_ID = 0

    numerator: int
    denominator: int


@dataclass
class ConcentratedPoolDatum(PoolDatum):
    """On-chain datum for a Danogo Concentrated Liquidity pool UTXO.

    Attributes:
        token_x: ``[policy_id, asset_name]`` bytes for the first token (X).
        token_y: ``[policy_id, asset_name]`` bytes for the second token (Y).
        lp_fee_rate: LP fee rate in basis points (e.g. 30 = 0.30 %).
        platform_fee_x: Accumulated platform fee in token X (lovelace when X
            is ADA).
        platform_fee_y: Accumulated platform fee in token Y.
        total_swap_fee: Running total of all swap fees collected, in token X.
        sqrt_lower_price: Rational ``sqrt(Pa)`` — square root of the lower
            price bound.
        sqrt_upper_price: Rational ``sqrt(Pb)`` — square root of the upper
            price bound.
        min_x_change: Minimum token-X change enforced by the contract.
        min_y_change: Minimum token-Y change enforced by the contract.
        circulating_lp_token: Total circulating LP tokens for this pool.
        last_withdraw_epoch: Epoch at which staking rewards were last
            withdrawn; used to gate reward-withdrawal logic.
    """

    CONSTR_ID = 0
    token_x: list[bytes]
    token_y: list[bytes]
    lp_fee_rate: Basis
    platform_fee_x: int
    platform_fee_y: int
    total_swap_fee: int
    sqrt_lower_price: DanogoPRational
    sqrt_upper_price: DanogoPRational
    min_x_change: int
    min_y_change: int
    circulating_lp_token: int
    last_withdraw_epoch: int

    def pool_pair(self) -> Assets | None:
        """Return the zero-quantity asset pair (X, Y) for this pool.

        Returns an ``Assets`` dict keyed by both token unit strings with
        quantities of 0, suitable for identifying which tokens the pool
        trades without implying any specific amounts.
        """
        asset_x = DanogoTupleAsset.from_list(self.token_x)
        asset_y = DanogoTupleAsset.from_list(self.token_y)
        return asset_x.assets + asset_y.assets


@dataclass
class ConcentratedOrderDatum(OrderDatum):
    """No-op order datum for Danogo Concentrated Liquidity pools.

    Danogo CL pools execute swaps directly against the pool UTXO; there is
    no separate order contract or order datum.  This stub class exists solely
    to satisfy the ``AbstractPairState.order_datum_class()`` interface
    contract.  All methods return deterministic placeholder values.
    """

    CONSTR_ID: int = 0

    @classmethod
    def create_datum(
        cls,
        address_source: Address,
        in_assets: Assets,
        out_assets: Assets,
        batcher_fee: Assets,
        deposit: Assets,
        address_target: Address | None = None,
        datum_target: PlutusData | None = None,
    ) -> "ConcentratedOrderDatum":
        """Return an empty ``ConcentratedOrderDatum`` instance.

        All parameters are accepted for API compatibility with
        ``AbstractPairState`` but are intentionally ignored — Danogo pools
        do not use an order datum.
        """
        return cls()

    def address_source(self) -> Address:
        """Return a stub source address (all-zeroes 28-byte payment key).

        Danogo pools have no real order datum, so there is no actual source
        address to recover.  The returned address is a deterministic,
        zero-filled placeholder and must never be used to sign or receive
        funds.
        """
        return Address(
            payment_part=VerificationKeyHash(b"\x00" * 28),
            network=Network.MAINNET,
        )

    def requested_amount(self) -> Assets:
        """Return an empty ``Assets`` dict (no requested amount).

        Danogo pools do not use an order datum, so requested amounts are not
        encoded on-chain.  This stub satisfies the interface contract.
        """
        return Assets({})

    def order_type(self) -> OrderType:
        """Return ``OrderType.swap`` — all Danogo CL interactions are direct swaps."""
        return OrderType.swap


# =============================================================================
# 2. INFRASTRUCTURE DATACLASSES
#    Plain data containers for pool-registry entries and script-reference info.
# =============================================================================


@dataclass
class Delegation:
    """Current staking delegation state for a Cardano stake address.

    Attributes:
        pool_id: Bech32 identifier of the pool the address is delegated to,
            or ``None`` if not currently delegated.
        rewards: Withdrawable staking rewards in lovelace (``0`` if not
            delegated or if Blockfrost is unreachable).
    """

    pool_id: str | None
    rewards: int


@dataclass
class ConcentratedPoolInfo:
    """Static deploy-time configuration for a single Danogo Concentrated Liquidity pool.

    Attributes:
        nft_unit: Full asset unit (``policy_hex + asset_name_hex``) of the
            pool's unique identifying NFT.
        address: Bech32 pool UTXO address.
        staking_tx_hash: Hex-encoded transaction ID that holds the staking
            script reference UTXO.
        staking_tx_index: Output index within ``staking_tx_hash``.
        staking_script_hash: Hex-encoded payment-script hash of the staking
            validator; used to derive the pool's reward address.
    """

    nft_unit: str
    address: str
    staking_tx_hash: str
    staking_tx_index: int
    staking_script_hash: str


@dataclass
class ConcentratedPoolScriptInfo:
    """Reference-input location and hash for the pool validator script.

    Attributes:
        out_ref: ``TransactionInput`` pointing to the UTXO that holds the
            pool script as an inline reference script.
        script_hash: Hex-encoded hash of the pool validator script.
    """

    out_ref: TransactionInput
    script_hash: str


@dataclass
class ConcentratedStakingScriptInfo:
    """Reference-input location and hash for the pool's staking validator.

    Attributes:
        out_ref: ``TransactionInput`` pointing to the UTXO that holds the
            staking script as an inline reference script.
        script_hash: Hex-encoded hash of the staking validator script;
            also used to derive the pool's reward address.
    """

    out_ref: TransactionInput
    script_hash: str


# =============================================================================
# 3. PURE MATH & UTILITY FUNCTIONS
#    Stateless, deterministic helpers.  Defined here so every call site below
#    has its dependency already resolved when you read top-to-bottom.
# =============================================================================


def ceil_div(a: int, b: int) -> int:
    """Return the ceiling of ``a / b`` using integer arithmetic only.

    Equivalent to ``math.ceil(a / b)`` but avoids floating-point conversion,
    which is important for the large integers that appear in CLMM pool math.

    Args:
        a: Dividend.
        b: Divisor (must be non-zero).

    Returns:
        Smallest integer ``n`` such that ``n * b >= a``.

    Raises:
        ValueError: If ``b`` is zero.
    """
    if b == 0:
        raise ValueError("Division by zero")

    # In DEX math, quantities are positive.
    # This is the Pythonic way to do ceiling division for positive integers:
    return (a + b - 1) // b


def get_epoch(t: int, network: Network) -> int:
    """Return the Cardano epoch number for a POSIX millisecond timestamp.

    Uses the Shelley era epoch boundary (epoch 328) as the reference point.
    Epoch length is 5 days (432 000 000 ms) on mainnet and 30 minutes
    (1 800 000 ms) on testnets.

    Args:
        t: Current time as a POSIX timestamp in **milliseconds**.
        network: Cardano network (``Network.MAINNET`` or a testnet variant).

    Returns:
        Integer epoch number corresponding to timestamp ``t``.
    """
    epoch_length = 432_000_000
    epoch_boundary = 1647899091000
    epoch_boundary_as_epoch = 328

    if network != Network.MAINNET:
        epoch_length = 1_800_000

    # Python's // is floor division, matching Math.floor()
    return ((t - epoch_boundary) // epoch_length) + epoch_boundary_as_epoch


SHELLEY_START_SLOT = 4492800
SHELLEY_START_TIME = 1596059091


def get_current_mainnet_slot() -> int:
    """Estimate the current Cardano Mainnet slot from the system clock.

    Uses the Shelley era genesis constants (slot 4 492 800 at Unix time
    1 596 059 091) to derive the slot without any network call.  Suitable
    as a fallback when the Blockfrost API is unavailable or rate-limited.

    Returns:
        Approximate current slot number.  May deviate from the true on-chain
        value by a few seconds due to clock skew or NTP drift.
    """
    # Get the current Unix timestamp (seconds since 1970)
    current_unix_time = int(time.time())

    # Calculate the current slot
    return SHELLEY_START_SLOT + (current_unix_time - SHELLEY_START_TIME)


def calc_liquidity(
    x: int,
    y: int,
    pa: tuple[int, int],
    pb: tuple[int, int],
) -> tuple[int, int]:
    """Compute the CLMM liquidity invariant ``L`` as a rational ``(num, den)``.

    Args:
        x: Active reserve of token X (after fee deductions).
        y: Active reserve of token Y (after fee deductions).
        pa: Square root of the lower price bound as ``(numerator, denominator)``.
        pb: Square root of the upper price bound as ``(numerator, denominator)``.

    Returns:
        ``(numerator, denominator)`` representing ``L`` as an exact fraction.
    """
    den_a_den_b = pa[1] * pb[1]
    num_a_num_b = pa[0] * pb[0]

    diff_square = (y * den_a_den_b - x * num_a_num_b) ** 2
    xy4_term = 4 * x * y * (pa[1] ** 2) * (pb[0] ** 2)

    # math.isqrt is highly optimized for huge integers in Python
    big_sqrt_in_numerator = math.isqrt(diff_square + xy4_term)

    numerator = y * den_a_den_b + x * num_a_num_b + big_sqrt_in_numerator
    denominator = 2 * (pb[0] * pa[1] - pb[1] * pa[0])

    return (numerator, denominator)


def calculate_l(
    x: Fraction,
    y: Fraction,
    sqrt_pa: Fraction,
    sqrt_pb: Fraction,
) -> Fraction:
    """Compute L = LNum/LDen.

    Args:
        x: Token X amount excluding fees.
        y: Token Y amount excluding fees.
        sqrt_pa: Square root of lower price bound.
        sqrt_pb: Square root of upper price bound.

    Returns:
        The liquidity value L as a Fraction.
    """
    p_a = sqrt_pa * sqrt_pa
    p_b = sqrt_pb * sqrt_pb

    if p_a <= 0 or p_b <= 0:
        raise ValueError("Price values must be positive")

    # t1 = X * sqrt(P_a) * sqrt(P_b)  # noqa: ERA001
    t1 = x * sqrt_pa * sqrt_pb

    # tmp = (Y - X*sqrt(P_a) * sqrt(P_b))  # noqa: ERA001
    tmp = y - t1
    tmp_squared = tmp * tmp

    # fXY = 4*X*Y*P_b  # noqa: ERA001
    f_xy = Fraction(4) * x * y * p_b

    # tmp_squared = (Y - X*sqrt(P_a) * sqrt(P_b))^2 + 4*X*Y*P_b  # noqa: ERA001
    tmp_squared = tmp_squared + f_xy

    product = tmp_squared.numerator * tmp_squared.denominator
    t2 = Fraction(math.isqrt(product), tmp_squared.denominator)

    # num = Y + t1 + t2  # noqa: ERA001
    num = y + t1 + t2

    # denom = 2 * (sqrt(P_b) - sqrt(P_a))  # noqa: ERA001
    denom_diff = sqrt_pb - sqrt_pa
    if denom_diff == 0:
        raise ValueError("Price range (Pb - Pa) cannot be zero")

    denom = Fraction(2) * denom_diff

    # L = num / denom  # noqa: ERA001
    return num / denom


def calculate_xv_yv(
    x: int,
    y: int,
    sqrt_pa_num: int,
    sqrt_pa_den: int,
    sqrt_pb_num: int,
    sqrt_pb_den: int,
) -> tuple[Fraction, Fraction]:
    """Compute the virtual reserves ``(Xv, Yv)`` for the current price range.

    Virtual reserves extend the real reserves with imaginary liquidity outside
    the current tick range so that the standard constant-product formula can
    be applied within a CLMM pool.  Derived as::

        Xv = ceil(L / sqrt(Pb)) + X
        Yv = ceil(L * sqrt(Pa)) + Y

    Each result is rounded up (ceiling) to avoid underestimating pool depth,
    which could allow invalid swap outputs.

    Args:
        x: Real reserve of token X.
        y: Real reserve of token Y.
        sqrt_pa_num: Numerator of the rational ``√(Pa)``.
        sqrt_pa_den: Denominator of the rational ``√(Pa)``.
        sqrt_pb_num: Numerator of the rational ``√(Pb)``.
        sqrt_pb_den: Denominator of the rational ``√(Pb)``.

    Returns:
        A ``(Xv, Yv)`` tuple where each element is a ceiling-rounded
        ``Fraction``.

    Raises:
        ValueError: If either price-bound denominator is zero.
    """
    try:
        sqrt_pa = Fraction(sqrt_pa_num, sqrt_pa_den)
        sqrt_pb = Fraction(sqrt_pb_num, sqrt_pb_den)

        l_value = calculate_l(Fraction(x), Fraction(y), sqrt_pa, sqrt_pb)

        # Virtual reserves: xv = L / sqrt(Pb), yv = L * sqrt(Pa)
        xv = l_value / sqrt_pb
        yv = l_value * sqrt_pa

        return Fraction(math.ceil(xv)), Fraction(math.ceil(yv))
    except ZeroDivisionError as err:
        raise ValueError("Invalid price bounds: Denominator cannot be zero.") from err


def get_pool_change(
    amount_in: int,
    token_in_virtual: int,
    token_out_virtual: int,
    token_out_real: int,
    lp_fee_rate: int,
) -> tuple[int, int]:
    """Apply the constant-product formula to derive output amount and platform fee.

    Args:
        amount_in: Gross input amount (before any fee split).
        token_in_virtual: Virtual reserve of the input token (Xv or Yv).
        token_out_virtual: Virtual reserve of the output token.
        token_out_real: Actual (real) reserve of the output token; safety
            guard that prevents draining more than the pool holds.
        lp_fee_rate: LP fee in basis points (e.g. 30 = 0.30 %).

    Returns:
        ``(expected_out, platform_fee)`` both in the output token's smallest
        unit.

    Raises:
        ValueError: If the computed output exceeds the real output reserve
            (``"pool out exceeded"``), indicating invalid pool state or input.
    """
    base = 10_000

    # fee calculations
    lp_fee = (amount_in * lp_fee_rate) // base
    platform_fee = (lp_fee * 5) // 100  # 5%
    off_fee = base - lp_fee_rate

    # main math
    denominator = token_in_virtual * base + amount_in * off_fee
    virtual_product = token_in_virtual * token_out_virtual

    numerator = token_out_virtual * denominator - virtual_product * base
    expected_out = numerator // denominator

    # safety check
    if expected_out > token_out_real:
        raise ValueError("pool out exceeded")

    return (expected_out, platform_fee)


def calculate_concentrated_pool_swap(
    token_a_amount: int,
    token_b_amount: int,
    datum: ConcentratedPoolDatum,
    delta_amount: int,
    reward_amount: int = 0,
) -> tuple[int, int]:
    """Compute the swap output and platform fee for a Danogo CLMM pool.

    Top-level entry point for swap math.  The calculation pipeline is:

    1. Strip platform fees and the MIN-ADA reserve to obtain *active* reserves.
    2. Compute the liquidity invariant ``L`` from active reserves and price bounds.
    3. Derive virtual reserves ``(Xv, Yv)`` from ``L``.
    4. Apply ``get_pool_change`` (constant-product formula) for the final output.

    **Swap direction** is encoded by the sign of ``delta_amount``:

    - **Positive** → token A (X) is being sold; token B (Y) is received.
    - **Negative** → token B (Y) is being sold; token A (X) is received.

    Args:
        token_a_amount: On-chain balance of token A (X) as read from the pool
            UTXO, in lovelace or the token's smallest unit.
        token_b_amount: On-chain balance of token B (Y), in its smallest unit.
        datum: Current ``ConcentratedPoolDatum`` decoded from the pool UTXO.
        delta_amount: Signed swap amount (positive = sell A, negative = sell B).
        reward_amount: Withdrawable staking rewards in lovelace to include in
            the active X reserve before pricing (default: 0).

    Returns:
        ``(expected_out, platform_fee)`` tuple in the output token's smallest
        unit.
    """
    pool_in_amount = -delta_amount if delta_amount < 0 else delta_amount

    # Check if Token X is Lovelace (Empty unit means ADA)
    min_ada = 3_000_000 if DanogoTupleAsset.from_list(datum.token_x).unit == "" else 0

    active_reserve_x = token_a_amount - datum.platform_fee_x + reward_amount - min_ada
    active_reserve_y = token_b_amount - datum.platform_fee_y

    pa = (datum.sqrt_lower_price.numerator, datum.sqrt_lower_price.denominator)
    pb = (datum.sqrt_upper_price.numerator, datum.sqrt_upper_price.denominator)

    liquidity = calc_liquidity(active_reserve_x, active_reserve_y, pa, pb)

    x_v = (
        ceil_div(
            liquidity[0] * datum.sqrt_upper_price.denominator,
            liquidity[1] * datum.sqrt_upper_price.numerator,
        )
        + active_reserve_x
    )

    y_v = (
        ceil_div(
            liquidity[0] * datum.sqrt_lower_price.numerator,
            liquidity[1] * datum.sqrt_lower_price.denominator,
        )
        + active_reserve_y
    )

    if delta_amount > 0:
        return get_pool_change(
            pool_in_amount,
            x_v,
            y_v,
            active_reserve_y,
            datum.lp_fee_rate,
        )

    return get_pool_change(
        pool_in_amount,
        y_v,
        x_v,
        active_reserve_x,
        datum.lp_fee_rate,
    )


def create_swap_redeemer_bytes(pool_in_idx: int, delta_amount: int) -> bytes:
    """Serialise the swap redeemer into a 36-byte Plutus data blob.

    Byte layout (36 bytes total, big-endian):

    +--------+------+-----------+-----------+------------------------------+
    | Offset | Size | Field     | Value     | Description                  |
    +========+======+===========+===========+==============================+
    |      0 |  1 B | pool_in   | variable  | Pool input UTxO index        |
    |      1 |  1 B | action    | 3         | Swap action tag              |
    |      2 |  1 B | pool_in   | variable  | Repeated (contract layout)   |
    |      3 |  1 B | pool_out  | 0         | Pool output index            |
    |   4-35 | 32 B | amount    | signed    | Delta amount, big-endian     |
    +--------+------+-----------+-----------+------------------------------+

    Args:
        pool_in_idx: Zero-based index of the pool input UTxO within the
            transaction's input list.
        delta_amount: Signed swap amount (positive = sell token A, negative =
            sell token B).  Must fit in a signed 32-byte integer.

    Returns:
        36 bytes ready to be wrapped in a ``Redeemer``.
    """
    swap_action = 3
    pool_out_idx = 0

    # Pack integers into big-endian bytes
    pool_in_bytes = pool_in_idx.to_bytes(1, byteorder="big")
    action_bytes = swap_action.to_bytes(1, byteorder="big")
    pool_out_bytes = pool_out_idx.to_bytes(1, byteorder="big")

    # 32 bytes for the amount.
    # signed=True allows for negative amounts if your logic requires it.
    amount_bytes = delta_amount.to_bytes(32, byteorder="big", signed=True)

    # Concatenate to match the TS logic exactly (Total 36 bytes)
    return pool_in_bytes + action_bytes + pool_in_bytes + pool_out_bytes + amount_bytes


def reward_address_from_script_hash(
    script_hash: ScriptHash | str,
    network: Network,
) -> Address:
    """Derive a Cardano staking reward address from a script hash.

    Args:
        script_hash: A ``ScriptHash`` object, or its lower-case hex string.
        network: ``Network.MAINNET`` or a testnet; controls the address
            network tag.

    Returns:
        An ``Address`` with ``payment_part=None`` and ``staking_part`` set
        to the provided script hash.
    """
    # Convert input to ScriptHash object if it's a hex string
    sh = (
        script_hash
        if isinstance(script_hash, ScriptHash)
        else ScriptHash.from_primitive(bytes.fromhex(script_hash))
    )

    return Address(
        payment_part=None,
        staking_part=sh,
        network=network,
    )


def reward_address_from_script(
    script: PlutusV1Script | PlutusV2Script | PlutusV3Script | str,
    network: Network,
) -> Address:
    """Derive a Cardano staking reward address from a compiled Plutus script.

    Args:
        script: A compiled ``PlutusV1Script``, ``PlutusV2Script``, or
            ``PlutusV3Script`` object, **or** the CBOR-encoded script bytes
            as a lower-case hex string.
        network: ``Network.MAINNET`` or a testnet.

    Returns:
        An ``Address`` whose staking credential is the script hash, suitable
        for querying rewards via the Blockfrost API.

    Raises:
        TypeError: If ``script`` is neither a ``PlutusScript`` object nor a
            ``str``.
        ValueError: If ``script`` is a hex string that cannot be decoded as
            any known Plutus version.
    """
    if isinstance(script, (PlutusV1Script, PlutusV2Script, PlutusV3Script)):
        script_obj = script

    elif isinstance(script, str):
        raw_bytes = bytes.fromhex(script)
        script_obj = None

        for script_class in [PlutusV3Script, PlutusV2Script, PlutusV1Script]:
            try:
                script_obj = script_class(raw_bytes)
                break
            except (ValueError, TypeError) as e:
                logger.debug(
                    "Script class %s rejected raw bytes: %s",
                    script_class.__name__,
                    e,
                )
                continue
    else:
        raise TypeError(
            "Input must be a specific PlutusV1/V2/V3Script object or a hex string.",
        )

    if script_obj is None:
        raise ValueError(
            "Provided script does not match any known Plutus version/format.",
        )

    script_hash = plutus_script_hash(script_obj)
    return reward_address_from_script_hash(script_hash, network)


# =============================================================================
# 4. BLOCKFROST BACKEND I/O
#    Network calls with TTL caching.  Depends on get_delegation_at above.
# =============================================================================


def get_delegation_at(stake_address: str) -> Delegation:
    """Fetch staking delegation info for a stake address via Blockfrost.

    Args:
        stake_address: Bech32-encoded stake address (e.g. ``stake1...``).

    Returns:
        A ``Delegation`` instance with the delegated pool ID and withdrawable
        rewards in lovelace.  Both fields are ``None`` / ``0`` when the
        address is not delegated.

    Raises:
        blockfrost.ApiError: For any Blockfrost error other than HTTP 404
            (address not found), which is silently converted to a
            zero-rewards delegation.
    """
    if "BLOCKFROST_PROJECT_ID" not in os.environ:
        return Delegation(pool_id=None, rewards=0)

    api = BlockFrostApi(project_id=os.environ["BLOCKFROST_PROJECT_ID"])

    try:
        account = api.account(stake_address=stake_address)

        # Return an instance of the Delegation class
        return Delegation(
            pool_id=account.pool_id,
            rewards=int(account.withdrawable_amount),
        )

    except ApiError as e:
        http_not_found = 404
        if e.status_code == http_not_found:
            return Delegation(pool_id=None, rewards=0)
        raise e from e


def get_current_slot() -> int:
    """Fetch the latest slot number from the Blockfrost API.

    Returns:
        The slot number of the most recent block seen by Blockfrost.

    Raises:
        OSError: If ``BLOCKFROST_PROJECT_ID`` is not set.
        blockfrost.ApiError: On any Blockfrost network or API error.
    """
    if "BLOCKFROST_PROJECT_ID" not in os.environ:
        raise OSError(
            "BLOCKFROST_PROJECT_ID not set in environment variables.",
        )

    api = BlockFrostApi(project_id=os.environ["BLOCKFROST_PROJECT_ID"])
    latest_block = api.block_latest()
    return latest_block.slot


REWARD_CACHE = TTLCache(maxsize=100, ttl=300)


@cached(cache=REWARD_CACHE)
def fetch_reward_from_blockfrost(staking_address: str) -> int:
    """Return the withdrawable staking rewards for a stake address (TTL-cached).

    Args:
        staking_address: Bech32-encoded stake address (e.g. ``stake1...``).

    Returns:
        Withdrawable rewards in lovelace, or ``0`` on any error (network
        unavailable, address not registered, Blockfrost rate-limited, etc.).
    """
    try:
        delegation_info = get_delegation_at(staking_address)
        return delegation_info.rewards if delegation_info else 0
    except (ApiError, OSError) as e:
        logger.warning("Failed to fetch reward for %s: %s", staking_address, e)
        return 0


# =============================================================================
# 5. CONCENTRATED POOL STATE
#    The main pydantic model representing a live on-chain pool UTXO.
# =============================================================================


class ConcentratedPoolState(AbstractPoolState):
    """Live on-chain state for a single Danogo Concentrated Liquidity pool UTXO."""

    address: str

    # CENTRAL REGISTRY: Hardcode the Danogo Concentrated Liquidity
    # pools and their configurations.
    _SUPPORTED_POOLS: ClassVar[list[ConcentratedPoolInfo]] = [
        ConcentratedPoolInfo(
            nft_unit=(
                "bafb08d9c4ed68d1730ed655c2ce5730c8abe2b9efd3eb237d9362c1365d"
                "a7a23e2c27bac8dbb06109804eedcf770bbff465304b4f30cf0b9c3e9ca8"
            ),
            address=(
                "addr_test1xza0kzxecnkk35tnpmt9tskw2ucv32lzh8ha86er0kfk9sfkdyf"
                "pf0tyqjnwaqeezz4h95tgt6dnmjla480lr0k0x8pstnjewc"
            ),
            staking_tx_hash=(
                "53a3234878cd92399fca30d0d941ebc9" "98080c9dc4c779296b77ae422a188898"
            ),
            staking_tx_index=1,
            staking_script_hash=(
                "36691214bd6404a6ee833910ab72d168" "5e9b3dcbfda9dff1becf31c3"
            ),
        ),
        ConcentratedPoolInfo(
            nft_unit=(
                "bafb08d9c4ed68d1730ed655c2ce5730c8abe2b9efd3eb237d9362c15c14"
                "f4012218e283c748c17532dcf211a7ac849705739189e7cfc4bea780e9fe"
            ),
            address=(
                "addr_test1xza0kzxecnkk35tnpmt9tskw2ucv32lzh8ha86er0kfk9swata"
                "27tgfwvfjrtuathj54c9h02sx5ytzat62f4cyxyjjsq80p3m"
            ),
            staking_tx_hash=(
                "14ab5d1bac6748958a83c080944a64cf" "7381030a37b22c02ce01da7a7d1715ac"
            ),
            staking_tx_index=1,
            staking_script_hash=(
                "dd5f55e5a12e626435f3abbca95c16ef" "540d422c5d5e949ae08624a5"
            ),
        ),
        ConcentratedPoolInfo(
            nft_unit=(
                "bafb08d9c4ed68d1730ed655c2ce5730c8abe2b9efd3eb237d9362c1cef8"
                "0e0978314255e268caa437056febc2be47e17fa914dd98155b62f5f3ba3a"
            ),
            address=(
                "addr_test1xza0kzxecnkk35tnpmt9tskw2ucv32lzh8ha86er0kfk9sfj6z"
                "lvm8uucky8guq7rtjjcr8cn5nwkfus5yrsefwv4tcsnkgzhy"
            ),
            staking_tx_hash=(
                "89d1ae5808d9d05b6d866ff0b4f14089" "e5dc9ed38514a69c9b142738e34355f8"
            ),
            staking_tx_index=1,
            staking_script_hash=(
                "32d0becd9f9cc58874701e1ae52c0cf8" "9d26eb2790a1070ca5ccaaf1"
            ),
        ),
    ]

    _stake_address: ClassVar[list[Address]] = [
        Address.from_primitive(pool.address) for pool in _SUPPORTED_POOLS
    ]

    _pool_nft_to_stake_address: ClassVar[dict[str, Address]] = {
        pool.nft_unit: Address.from_primitive(pool.address) for pool in _SUPPORTED_POOLS
    }

    _pool_nft_to_staking_script_info: ClassVar[
        dict[str, ConcentratedStakingScriptInfo]
    ] = {
        pool.nft_unit: ConcentratedStakingScriptInfo(
            out_ref=TransactionInput(
                transaction_id=TransactionId(bytes.fromhex(pool.staking_tx_hash)),
                index=pool.staking_tx_index,
            ),
            script_hash=pool.staking_script_hash,
        )
        for pool in _SUPPORTED_POOLS
    }

    _pool_script_info: ClassVar[
        ConcentratedPoolScriptInfo
    ] = ConcentratedPoolScriptInfo(
        out_ref=TransactionInput(
            transaction_id=TransactionId(
                bytes.fromhex(
                    "2e19cca74e3badcab26aef7574aa1885ba97228a254ca227ba2f79f2b75fd136",
                ),
            ),
            index=0,
        ),
        script_hash="04041c3c6ba87b33f2c9eb7f7dbeae3b26003c3e199d438bb99932a2",
    )

    @classmethod
    def dex(cls) -> str:
        """Return the human-readable name of this DEX implementation."""
        return "Danogo Concentrated Liquidity"

    @classmethod
    def order_selector(cls) -> list[str]:
        """Return Bech32-encoded pool addresses used to identify Danogo UTxOs.

        Danogo Concentrated Liquidity pools embed swaps directly into the pool
        UTXO rather than posting separate order UTxOs, so the “order” selector
        addresses are identical to the pool addresses.
        """
        return [s.encode() for s in cls._stake_address]

    @classmethod
    def pool_selector(cls) -> PoolSelector:
        """Return the ``PoolSelector`` criteria the Factory uses to find Danogo pools.

        The selector matches both by pool address and by the presence of the
        unique pool NFT, ensuring only genuine Danogo CL pool UTxOs are
        returned by backend UTXO queries.
        """
        return PoolSelector(
            # TODO: Replace with actual Danogo CL Pool script addresses
            addresses=cls.order_selector(),
            # TODO: Replace with actual Danogo CL Pool NFT Policy IDs
            assets=[pool.nft_unit for pool in cls._SUPPORTED_POOLS],
        )

    @classmethod
    def order_datum_class(cls) -> type[ConcentratedOrderDatum]:
        """Return the stub order-datum class for this DEX.

        Danogo pools have no real order datum; ``ConcentratedOrderDatum`` is
        an empty placeholder that satisfies the ``AbstractPairState`` interface.
        """
        return ConcentratedOrderDatum

    @classmethod
    def pool_datum_class(cls) -> type[ConcentratedPoolDatum]:
        """Return the on-chain pool datum class used for CBOR (de)serialisation."""
        return ConcentratedPoolDatum

    @classmethod
    def default_script_class(
        cls,
    ) -> type[PlutusV1Script] | type[PlutusV2Script] | type[PlutusV3Script]:
        """Return ``PlutusV3Script`` as the default Plutus version for this DEX."""
        return PlutusV3Script

    @classmethod
    def script_class(cls) -> type[PlutusV2Script] | type[PlutusV3Script]:
        """Return ``PlutusV3Script`` as the Plutus script version for this DEX."""
        return PlutusV3Script

    @classmethod
    def extract_pool_nft(cls, values: dict[str, Any]) -> Assets | None:
        """Extract the unique pool NFT from a pool UTXO's asset map.

        Danogo CL pool UTxOs carry exactly one quantity-1 token that acts as
        the pool identifier.  This method locates that NFT, removes it from
        the liquid assets dict (``values["assets"]``), and stores it under
        ``values["pool_nft"]`` for downstream parsing.

        If ``pool_nft`` is already present in ``values`` (e.g. the UTXO has
        been parsed before), the existing value is returned immediately.

        Args:
            values: Raw pool UTXO field dict as produced by the backend.

        Returns:
            An ``Assets`` dict containing the single pool NFT unit and its
            quantity (always 1 for a valid pool), or ``None`` if not found.

        Raises:
            InvalidPoolError: If the UTXO does not contain exactly one
                quantity-1 token.
        """
        assets = values["assets"]

        if "pool_nft" in values:
            pool_nft = Assets(root=values["pool_nft"])
        else:
            nfts = [asset for asset, quantity in assets.items() if quantity == 1]
            if len(nfts) != 1:
                raise InvalidPoolError(
                    f"Concentrated pools must have exactly one pool nft:"
                    f" assets={assets}",
                )
            pool_nft = Assets(**{nfts[0]: assets.root.pop(nfts[0])})
            values["pool_nft"] = pool_nft

        return pool_nft

    @property
    def pool_id(self) -> str:
        """Return the unique pool identifier (pool NFT's full asset unit)."""
        return self.pool_nft.unit()

    @property
    def swap_forward(self) -> bool:
        """Return ``False`` — Danogo pools do not support swap forwarding."""
        return False

    @property
    def stake_address(self) -> Address:
        """Return the pool's stake address, looked up by pool NFT unit.

        For Danogo CL pools the stake address doubles as the pool address
        (there is no separate order contract).  Falls back to the first
        registered pool address if the NFT unit is not found in the registry.
        """
        return self._pool_nft_to_stake_address.get(
            self.pool_nft.unit(),
            self._stake_address[0],
        )

    @computed_field
    @cached_property
    def pool_script_reference(self) -> ScriptReference:
        """Return the on-chain reference for the pool validator script.

        Resolved once via the configured backend and then cached on this
        instance.  Requires ``_pool_script_info.script_hash`` to be correct.
        """
        return get_backend().get_script_from_address(
            Address(
                payment_part=ScriptHash.from_primitive(
                    self._pool_script_info.script_hash,
                ),
            ),
        )

    @computed_field
    @cached_property
    def staking_script_reference(self) -> ScriptReference | None:
        """Return the on-chain reference for this pool's staking validator.

        Resolved once via the configured backend and then cached on this
        instance.  Returns ``None`` if no staking script is registered for
        this pool's NFT unit in ``_pool_nft_to_staking_script_info``.
        """
        staking_info = self._pool_nft_to_staking_script_info.get(self.pool_nft.unit())
        if staking_info is None:
            return None

        return get_backend().get_script_from_address(
            Address(payment_part=ScriptHash.from_primitive(staking_info.script_hash)),
        )

    @computed_field
    def pool_reward_amount(self) -> int:
        """Return the withdrawable staking reward for this pool in lovelace.

        Delegates to ``fetch_reward_from_blockfrost`` (TTL-cached) to avoid
        excessive Blockfrost API calls during rapid pool-state queries.
        Returns ``0`` if the staking script reference is not available or
        if any error occurs while resolving the reward address.
        """
        if self.staking_script_reference is None:
            return 0

        try:
            staking_reward_address = reward_address_from_script(
                self.staking_script_reference.script,
                Network.MAINNET,
            )
            address_str = staking_reward_address.encode()
            return fetch_reward_from_blockfrost(address_str)

        except (ValueError, TypeError) as e:
            logger.error(
                "Failed to resolve staking address for pool %s: %s",
                self.pool_nft.unit(),
                e,
            )
            return 0

    @property
    def price(self) -> tuple[Decimal, Decimal]:
        """Return spot prices derived from the pool's virtual reserves ``(Xv, Yv)``.

        Virtual reserves account for the CLMM tick structure and give a price
        that reflects the marginal rate at the current tick, rather than a
        simple ratio of raw on-chain balances.

        Returns:
            A ``(price_b_in_a, price_a_in_b)`` tuple of ``Decimal`` values
            where:

            - ``price_b_in_a`` = units of token A required to buy 1 token B
              (= Yv / Xv).
            - ``price_a_in_b`` = units of token B required to buy 1 token A
              (= Xv / Yv).

            Both values are ``Decimal(0)`` if either virtual reserve is zero.
        """
        # 1. Get actual real reserves from UTXO (normalized to natural units)
        nat_assets = naturalize_assets(self.assets)

        # 2. Calculate available real reserves (deducting platform and swap fees)
        # !!! Note: ignoring minAda deduction for now
        x_reserve = (
            Decimal(nat_assets.get(self.unit_a, 0))
            - Decimal(self.pool_datum.platform_fee_x)
            - Decimal(self.pool_datum.total_swap_fee)
        )

        y_reserve = Decimal(nat_assets.get(self.unit_b, 0)) - Decimal(
            self.pool_datum.platform_fee_y,
        )

        # Ensure reserves don't drop below zero due to fees
        x_reserve = max(x_reserve, Decimal(0))
        y_reserve = max(y_reserve, Decimal(0))

        # 3. Calculate Virtual Reserves (Xv, Yv) using the current tick's price bounds
        # Note: calculate_xv_yv returns ceiled Fraction objects.
        xv_frac, yv_frac = calculate_xv_yv(
            int(x_reserve),
            int(y_reserve),
            self.pool_datum.sqrt_lower_price.numerator,
            self.pool_datum.sqrt_lower_price.denominator,
            self.pool_datum.sqrt_upper_price.numerator,
            self.pool_datum.sqrt_upper_price.denominator,
        )

        # 4. Safe check before division to prevent ZeroDivisionError
        if xv_frac == 0 or yv_frac == 0:
            return (Decimal(0), Decimal(0))

        # 5. Convert ceiled Fractions to Decimals for the final price calculation
        xv = Decimal(int(xv_frac))
        yv = Decimal(int(yv_frac))

        # Price of B in units of A = Yv / Xv
        price_b_in_a = yv / xv

        # Price of A in units of B = Xv / Yv
        price_a_in_b = xv / yv

        return (price_b_in_a, price_a_in_b)

    @property
    def tvl(self) -> Decimal:
        """Return the Total Value Locked (TVL) denominated in ADA.

        For ADA-paired pools, the pool's ADA balance is doubled (assuming a
        50/50 value split) and converted from lovelace to ADA.

        Returns:
            TVL in ADA as a ``Decimal`` (lovelace / 1 000 000 * 2).

        Raises:
            NotImplementedError: For pools that do not contain ADA (e.g.
                USDM/XYZ), as cross-token price data is not yet available.
        """
        # 1. Normalize the assets in the UTXO to natural units (Decimal)
        nat_assets = naturalize_assets(self.assets)

        # 2. Get the actual Lovelace (ADA) amount present in the pool
        lovelace_amount = Decimal(nat_assets.get("lovelace", 0))

        # Pool contains ADA (e.g., ADA/USDM)
        # We assume the other token's value equals the ADA in the pool (50/50 split).
        if self.unit_a == "lovelace" or self.unit_b == "lovelace":
            # TVL = ADA Amount * 2
            # Divide by 1,000,000 to convert from Lovelace to ADA
            return (lovelace_amount * Decimal(2)) / Decimal(1_000_000)

        # Pool does not contain ADA (e.g., USDM/XYZ)
        msg = "tvl for non-ADA pools is not implemented."
        raise NotImplementedError(msg)

    def _get_amount_out_and_platform_fee(
        self,
        asset: Assets,
        reward_amount: int,
    ) -> tuple[Assets, int]:
        """Calculate the swap output and platform fee for a given input asset.

        Args:
            asset: A single-token ``Assets`` object representing the input
                (must contain exactly one token key).
            reward_amount: Withdrawable staking rewards in lovelace to include
                in the active X reserve when pricing the swap.

        Returns:
            A ``(output_assets, platform_fee)`` tuple:

            - ``output_assets``: ``Assets`` with the received token and its
              quantity.
            - ``platform_fee``: Platform fee in lovelace (or the input token's
              smallest unit) accrued as a result of this swap.

        Raises:
            ValueError: If ``asset`` contains more than one token, or if the
                token is not one of the two pool tokens.
        """
        if len(asset) != 1:
            raise ValueError("Asset should only have one token.")

        pool_in_datum = ConcentratedPoolDatum.from_cbor(self.datum_cbor)

        token_a_unit = (
            DanogoTupleAsset.from_list(pool_in_datum.token_x).unit or "lovelace"
        )
        token_b_unit = DanogoTupleAsset.from_list(pool_in_datum.token_y).unit

        # Check that the input asset is either token A or token B in the pool
        if asset.unit() not in [token_a_unit, token_b_unit]:
            raise ValueError(
                "Input asset must be either token A or token B in the pool.",
            )

        # Determine the direction of the swap to set delta_amount
        # (positive if token A is input, negative if token B is input)
        delta_amount = (
            asset.quantity() if token_a_unit == asset.unit() else -asset.quantity()
        )

        # Calculate the amount to receive and platform fee based on the swap
        # direction, input amount, and current pool state
        token_to_receive_amount, platform_fee = calculate_concentrated_pool_swap(
            token_a_amount=self.assets[token_a_unit],
            token_b_amount=self.assets[token_b_unit],
            datum=pool_in_datum,
            delta_amount=delta_amount,
            reward_amount=reward_amount,
        )

        token_in_unit = asset.unit()
        token_out_unit = token_b_unit if token_in_unit == token_a_unit else token_a_unit

        return Assets(root={token_out_unit: int(token_to_receive_amount)}), platform_fee

    def get_amount_out(self, asset: Assets) -> tuple[Assets, float]:
        """Return the expected swap output for a given input amount.

        Fetches the current staking reward amount, then delegates to
        ``_get_amount_out_and_platform_fee`` for CLMM pricing.  Price impact
        is not calculated for Danogo CLMM pools and is always ``0.0``.

        Args:
            asset: A single-token ``Assets`` object (the input being sold).

        Returns:
            A ``(output_assets, price_impact)`` tuple.  ``price_impact`` is
            always ``0.0`` — use ``get_amount_in`` for slippage bounding.

        Raises:
            ValueError: If ``asset`` contains more than one token.
        """
        if len(asset) != 1:
            raise ValueError("Asset should only have one token.")

        reward_amount = self.pool_reward_amount
        amount_out, _ = self._get_amount_out_and_platform_fee(
            asset=asset,
            reward_amount=reward_amount,
        )

        price_impact = 0.0
        return amount_out, price_impact

    def _find_upper_bound(
        self,
        unit_in: str,
        low: int,
        target_out_amount: int,
        reward_amount: int,
    ) -> tuple[int, int]:
        """Grow ``high`` exponentially until output meets target; return (low, high).

        As successful trial inputs are found they are promoted to ``low`` so
        that the subsequent binary search starts from a tight lower bound.
        Stops early if the pool is exhausted (``ValueError`` from pricing).
        """
        high = int(low * 1.05) + 1
        for _ in range(_MAX_SEARCH_ITERATIONS):
            test_asset = Assets(**{unit_in: high})
            try:
                out_result, _ = self._get_amount_out_and_platform_fee(
                    test_asset,
                    reward_amount,
                )
                if out_result.quantity() >= target_out_amount:
                    break
                low = high
                high = int(high * 1.05) + 1
            except ValueError:
                break
        return low, high

    def _binary_search_min_input(
        self,
        unit_in: str,
        low: int,
        high: int,
        target_out_amount: int,
        reward_amount: int,
    ) -> int:
        """Bisect ``[low, high]`` to find the minimum input for ``target_out_amount``.

        Returns the smallest ``mid`` whose simulated output is at least
        ``target_out_amount``.  Inputs that exhaust the pool raise
        ``ValueError`` from the pricing function; they are treated as too
        large and the search narrows toward ``low``.
        """
        best_in = high
        while low <= high:
            mid = (low + high) // 2
            test_asset = Assets(**{unit_in: mid})
            try:
                out_result, _ = self._get_amount_out_and_platform_fee(
                    test_asset,
                    reward_amount,
                )
                if out_result.quantity() >= target_out_amount:
                    best_in = mid
                    high = mid - 1
                else:
                    low = mid + 1
            except ValueError:
                high = mid - 1
        return best_in

    def get_amount_in(self, asset: Assets) -> tuple[Assets, float]:
        """Return the minimum input needed to receive a desired output amount.

        Because CLMM liquidity is non-uniform across ticks, there is no
        closed-form inverse.  The method uses a two-phase search:

        1. **Seed estimate** - performs the reverse swap (output → input) to
           get a symmetric lower bound, accounting for slippage direction.
        2. **Exponential growth** - starting 5 % above the seed, the trial
           input is multiplied by 1.05 each iteration until the simulated
           output meets or exceeds the target (upper bound).
        3. **Binary search** - bisects ``[low, high]`` to find the smallest
           input that satisfies the target, with at most ``O(log N)`` pricing
           calls.
        4. **Final verification** - confirms the found input actually produces
           the required output before returning.

        Args:
            asset: A single-token ``Assets`` object representing the desired
                output and its quantity.

        Returns:
            A ``(input_assets, slippage_fee)`` tuple.  ``slippage_fee`` is
            always ``0.0`` (not yet computed for Danogo pools).

        Raises:
            ValueError: If the target output amount is not positive, or if
                pool liquidity is insufficient to fulfill the request.
        """

        target_out_amount = asset.quantity()
        unit_out = asset.unit()

        if target_out_amount <= 0:
            raise ValueError(
                f"Output amount must be positive. Got: {target_out_amount}",
            )

        token_x = DanogoTupleAsset.from_list(self.pool_datum.token_x).unit or "lovelace"
        token_y = DanogoTupleAsset.from_list(self.pool_datum.token_y).unit or "lovelace"

        is_x_out = unit_out == token_x
        unit_in = token_y if is_x_out else token_x

        # --- Step 1: Seed estimate ---
        # Feed the target output into the reverse direction to get a symmetric
        # starting guess that mirrors the current CLMM tick structure.
        guess_asset = Assets(**{unit_out: target_out_amount})
        try:
            guess_result_asset, _ = self._get_amount_out_and_platform_fee(
                guess_asset,
                self.pool_reward_amount,
            )
            guess_in = guess_result_asset.quantity()
        except ValueError:
            # If the reverse swap exhausts the pool, fall back to 1.
            guess_in = 1

        low = max(1, guess_in)

        # The reverse-swap seed can overshoot (fees compound in both directions),
        # meaning `low` itself may already produce >= target_out_amount.  If so,
        # halve low so the binary search has room to find a smaller true minimum.
        try:
            seed_out, _ = self._get_amount_out_and_platform_fee(
                Assets(**{unit_in: low}),
                self.pool_reward_amount,
            )
            if seed_out.quantity() >= target_out_amount:
                low = max(1, low // 2)
        except ValueError:
            pass

        # --- Steps 2 & 3: Exponential search then binary search ---
        low, high = self._find_upper_bound(
            unit_in, low, target_out_amount, self.pool_reward_amount,
        )
        best_in = self._binary_search_min_input(
            unit_in, low, high, target_out_amount, self.pool_reward_amount,
        )

        # --- Step 4: Final verification ---
        final_test_asset = Assets(**{unit_in: best_in})
        try:
            final_out, _ = self._get_amount_out_and_platform_fee(
                final_test_asset,
                self.pool_reward_amount,
            )
        except ValueError as e:
            raise ValueError(f"Cannot calculate amount_in: {e}") from e

        if final_out.quantity() < target_out_amount:
            raise ValueError("Insufficient liquidity to fulfill this target output.")

        return final_test_asset, 0.0

    def swap_datum(
        self,
        address_source: Address,
        in_assets: Assets,
        out_assets: Assets,
        extra_assets: Assets | None = None,
        address_target: Address | None = None,
        datum_target: PlutusData | None = None,
    ) -> PlutusData:
        """Return a no-op order datum for protocol-level interface compatibility.

        Danogo CL pools do not use a separate order datum; the swap is executed
        directly against the pool UTXO via ``swap_utxo``.  This method returns
        an empty ``ConcentratedOrderDatum`` to satisfy the ``AbstractPairState``
        interface.  Swap forwarding is not supported; a warning is logged if
        ``address_target`` is provided.

        Args:
            address_source: Sender's address (accepted but unused).
            in_assets: Input assets being sold (accepted but unused).
            out_assets: Expected output assets (accepted but unused).
            extra_assets: Additional assets (accepted but unused).
            address_target: Forward-swap target; triggers a warning if set.
            datum_target: Forward-swap datum (accepted but unused).

        Returns:
            An empty ``ConcentratedOrderDatum`` instance.
        """
        if self.swap_forward and address_target is not None:
            logger.warning(
                "%s does not support swap forwarding.",
                self.__class__.__name__,
            )

        return self.order_datum_class().create_datum(
            address_source=address_source,
            in_assets=in_assets,
            out_assets=out_assets,
            batcher_fee=Assets({}),
            deposit=Assets({}),
            address_target=address_target,
            datum_target=datum_target,
        )

    def swap_utxo(
        self,
        address_source: Address,
        in_assets: Assets,
        out_assets: Assets,
        tx_builder: TransactionBuilder | None = None,
        extra_assets: Assets | None = None,
        address_target: Address | None = None,
        datum_target: PlutusData | None = None,
    ) -> tuple[TransactionOutput | None, PlutusData]:
        """Build and register all transaction components for a Danogo direct swap.

        Args:
            address_source: The sender's address (used in the order datum).
            in_assets: Assets being sold into the pool.
            out_assets: Assets expected to be received from the pool.
            tx_builder: A ``TransactionBuilder`` to mutate; must not be
                ``None``.
            extra_assets: Unused; present for interface compatibility.
            address_target: Unused; swap forwarding is not supported.
            datum_target: Unused; present for interface compatibility.

        Returns:
            A ``(pool_output, new_pool_datum)`` tuple:

            - ``pool_output``: The ``TransactionOutput`` for the updated pool
              UTXO, ready to be submitted on-chain.
            - ``new_pool_datum``: The updated ``ConcentratedPoolDatum``
              reflecting the swap's effect on fees and epoch tracking.

        Raises:
            ValueError: If any required precondition is missing (``tx_hash``,
                ``pool_nft``, ``pool_script_reference``,
                ``staking_script_reference``, or ``tx_builder``).
        """
        if self.tx_hash is None:
            raise ValueError("Transaction hash is required for swap operation")
        if self.pool_nft is None:
            raise ValueError("Pool NFT is required for swap operation")
        if self.pool_script_reference is None:
            raise ValueError("Pool script reference is required for swap operation")
        if self.staking_script_reference is None:
            raise ValueError("Staking script reference is required for swap operation")
        if tx_builder is None:
            raise ValueError("Transaction builder is required for swap operation")

        logger.info("Preparing transaction for Danogo Direct Swap...")

        pool_in_datum = ConcentratedPoolDatum.from_cbor(self.datum_cbor)

        # Extract token units (A/B) from the pool datum
        token_a_unit = (
            DanogoTupleAsset.from_list(pool_in_datum.token_x).unit or "lovelace"
        )
        token_b_unit = DanogoTupleAsset.from_list(pool_in_datum.token_y).unit

        # Extract the input and output token units from the provided assets
        token_in_unit = in_assets.unit()
        token_out_unit = out_assets.unit()

        # Determine the direction of the swap to set delta_amount
        # (positive if token A is input, negative if token B is input)
        delta_amount = (
            in_assets.quantity()
            if token_a_unit == token_in_unit
            else -in_assets.quantity()
        )

        # Calculate the amount to receive and platform fee based on the swap
        # direction, input amount, and current pool state
        token_to_receive_amount, platform_fee = calculate_concentrated_pool_swap(
            token_a_amount=self.assets[token_a_unit],
            token_b_amount=self.assets[token_b_unit],
            datum=pool_in_datum,
            delta_amount=delta_amount,
            reward_amount=self.pool_reward_amount,
        )

        # Construct transaction builder with all necessary inputs, outputs,
        # and scripts for the swap operation

        # Add reference input: pool script
        tx_builder.reference_inputs.add(self._pool_script_info.out_ref)

        # Add reference input: staking script
        tx_builder.reference_inputs.add(
            self._pool_nft_to_staking_script_info.get(self.pool_nft.unit()).out_ref,
        )

        # Add input: pool_in
        swap_redeemer_bytes = create_swap_redeemer_bytes(
            pool_in_idx=0,
            delta_amount=delta_amount,
        )
        swap_redeemer = Redeemer(swap_redeemer_bytes)

        tx_builder.add_script_input(
            utxo=UTxO(
                input=TransactionInput(
                    transaction_id=TransactionId(bytes.fromhex(self.tx_hash)),
                    index=self.tx_index,
                ),
                output=TransactionOutput(
                    address=Address.decode(self.address),
                    amount=asset_to_value(self.assets),
                    datum=self.pool_datum,
                ),
            ),
            redeemer=swap_redeemer,
        )

        # Add withdrawals
        ## Zero Withdrawal
        pool_script_reward_address = reward_address_from_script(
            self.pool_script_reference.script,
            Network.MAINNET,
        )
        tx_builder.withdrawals = Withdrawals(
            {bytes(pool_script_reward_address): 0},
        )
        tx_builder.add_withdrawal_script(
            script=PlutusV3Script.fromhex(self.pool_script_reference.script),
            redeemer=swap_redeemer,
        )

        current_ms = int(time.time() * 1000)
        current_epoch = get_epoch(current_ms, Network.MAINNET)

        # If this is pool Ada input, and the pool has staking rewards to withdraw
        can_withdraw_rewards = (
            token_a_unit == "lovelace"
            and current_epoch > pool_in_datum.last_withdraw_epoch
        )
        if can_withdraw_rewards:
            staking_reward_address = reward_address_from_script(
                self.staking_script_reference.script,
                Network.MAINNET,
            )
            tx_builder.withdrawals[
                bytes(staking_reward_address)
            ] = self.pool_reward_amount
            tx_builder.add_withdrawal_script(
                script=PlutusV3Script.fromhex(self.staking_script_reference.script),
                redeemer=swap_redeemer,
            )

        # Add output: new pool UTXO with updated Datum reflecting the swap and fees
        new_pool_datum = replace(
            pool_in_datum,
            platform_fee_x=pool_in_datum.platform_fee_x
            + (platform_fee if token_in_unit == token_a_unit else 0),
            platform_fee_y=pool_in_datum.platform_fee_y
            + (platform_fee if token_in_unit == token_b_unit else 0),
            last_withdraw_epoch=current_epoch,
        )

        pool_out_assets = (
            self.assets
            + Assets(**{token_in_unit: abs(delta_amount)})
            - Assets(**{token_out_unit: token_to_receive_amount})
        )

        pool_output = TransactionOutput(
            address=Address.decode(self.address),
            amount=asset_to_value(pool_out_assets),
            datum=new_pool_datum,
        )

        # Add output
        tx_builder.add_output(pool_output)

        # --- 9. TRANSACTION METADATA & TTL ---
        try:
            current_slot = get_current_slot()
        except (ApiError, OSError) as e:
            logger.warning(
                "Failed to fetch current slot from Blockfrost: %s."
                " Using fallback TTL values.",
                e,
            )
            current_slot = get_current_mainnet_slot()

        tx_builder.validity_start = current_slot - 120
        tx_builder.ttl = current_slot + 240
        tx_builder.fee = 17000

        # Attach CIP-20 message metadata for easy identification
        # on Cardanoscan / Cexplorer.
        tx_metadata = Metadata({674: {"msg": ["Danogo Liquidity Pair: Swap"]}})
        tx_builder.auxiliary_data = AuxiliaryData(tx_metadata)

        return pool_output, new_pool_datum
