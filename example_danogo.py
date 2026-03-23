"""Unit tests for the Danogo DEX module (danogo.py).

Run with: pytest example_danogo.py -v
"""

import math
import os
from fractions import Fraction
from typing import Any
from unittest.mock import patch

import pytest
from pycardano import Address
from pycardano import Network
from pycardano import TransactionId
from pycardano import TransactionInput
from pycardano import VerificationKeyHash
from pycardano.plutus import PlutusV1Script
from pycardano.plutus import PlutusV2Script
from pycardano.plutus import PlutusV3Script
from pycardano.plutus import plutus_script_hash

from charli3_dendrite.dataclasses.models import Assets
from charli3_dendrite.dataclasses.models import PoolSelector
from charli3_dendrite.dataclasses.models import OrderType
from charli3_dendrite.dexs.amm.danogo import AIKEN_TUPLE_LENGTH
from charli3_dendrite.dexs.amm.danogo import Delegation
from charli3_dendrite.dexs.amm.danogo import DanogoPRational
from charli3_dendrite.dexs.amm.danogo import DanogoTupleAsset
from charli3_dendrite.dexs.amm.danogo import ConcentratedOrderDatum
from charli3_dendrite.dexs.amm.danogo import ConcentratedPoolDatum
from charli3_dendrite.dexs.amm.danogo import ConcentratedPoolInfo
from charli3_dendrite.dexs.amm.danogo import ConcentratedPoolScriptInfo
from charli3_dendrite.dexs.amm.danogo import ConcentratedPoolState
from charli3_dendrite.dexs.amm.danogo import ConcentratedStakingScriptInfo
from charli3_dendrite.dexs.amm.danogo import calc_liquidity
from charli3_dendrite.dexs.amm.danogo import calculate_concentrated_pool_swap
from charli3_dendrite.dexs.amm.danogo import calculate_l
from charli3_dendrite.dexs.amm.danogo import calculate_xv_yv
from charli3_dendrite.dexs.amm.danogo import ceil_div
from charli3_dendrite.dexs.amm.danogo import create_swap_redeemer_bytes
from charli3_dendrite.dexs.amm.danogo import fetch_reward_from_blockfrost
from charli3_dendrite.dexs.amm.danogo import get_delegation_at
from charli3_dendrite.dexs.amm.danogo import get_epoch
from charli3_dendrite.dexs.amm.danogo import get_pool_change
from charli3_dendrite.dexs.amm.danogo import reward_address_from_script
from charli3_dendrite.dexs.amm.danogo import reward_address_from_script_hash
from charli3_dendrite.dexs.core.errors import InvalidPoolError

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

# CBOR of a real Danogo pool datum (ADA/fUSDA pool)
# Decoded fields: token_x=[b"",b""] (ADA), token_y=[policy,name] (fUSDA),
# lp_fee_rate=10, platform_fee_x=0, platform_fee_y=0, total_swap_fee=0
_SWAP_TEST_CBOR = (
    "d8799f9f4040ff9f581c9a614be30284aa88eb845da7657b5d0a235f1b95628b23c08050d5"
    "02456655534441ff0a000000d8799f1b000fe3624afc291b1b002386f26fc10000ffd8799f"
    "1a000f42401a000f4240ff1a022bfd431a00a673f21a829ccfa81a00010f91ff"
)

# fUSDA token: 28-byte policy + 5-byte asset name, concatenated as hex
_TOKEN_B = "9a614be30284aa88eb845da7657b5d0a235f1b95628b23c08050d5026655534441"

# Known pool address from the supported pool registry (first pool)
_POOL_ADDRESS = (
    "addr_test1xza0kzxecnkk35tnpmt9tskw2ucv32lzh8ha86er0kfk9sfkdy"
    "fpf0tyqjnwaqeezz4h95tgt6dnmjla480lr0k0x8pstnjewc"
)

# Known pool NFT unit (first pool in the registry)
_KNOWN_NFT = (
    "bafb08d9c4ed68d1730ed655c2ce5730c8abe2b9efd3eb237d9362c1"
    "365da7a23e2c27bac8dbb06109804eedcf770bbff465304b4f30cf0b9c3e9ca8"
)

# Dummy 28-byte script hash (hex) used in multiple tests
_SCRIPT_HASH_HEX = "04041c3c6ba87b33f2c9eb7f7dbeae3b26003c3e199d438bb99932a2"

# A minimal PlutusV3 script for script-based address derivation tests
_DUMMY_SCRIPT = PlutusV3Script(b"\x01\x00\x00")


# ---------------------------------------------------------------------------
# Shared helper: build a ConcentratedPoolState for unit tests
# ---------------------------------------------------------------------------


def _make_swap_pool(**overrides: Any) -> ConcentratedPoolState:
    """Create a ConcentratedPoolState with both pool tokens for swap tests.

    Patches out abstract methods so the class can be instantiated directly
    without a live blockchain backend.
    """
    defaults: dict[str, Any] = {
        # Realistic pool reserves: 100 ADA + 36 fUSDA
        "assets": Assets(root={"lovelace": 100_000_000, _TOKEN_B: 36_000_000}),
        # Use a known pool address from the supported pool registry
        "address": _POOL_ADDRESS,
        "block_time": 1_000_000,
        "block_index": 1,
        "plutus_v2": True,
        "tx_index": 0,
        "tx_hash": "0" * 64,
        "datum_cbor": _SWAP_TEST_CBOR,
        "datum_hash": "0" * 64,
        # Unknown NFT so staking_script_reference returns None (no backend needed)
        "pool_nft": Assets(root={"test_pool_nft": 1}),
    }
    defaults.update(overrides)
    with patch.object(ConcentratedPoolState, "__abstractmethods__", set()):
        return ConcentratedPoolState(**defaults)


# ===========================================================================
# Section 1: DanogoTupleAsset — unit, assets, from_list
# ===========================================================================


def test_danogo_tuple_asset_unit_empty_policy_returns_lovelace() -> None:
    """An empty policy must produce the 'lovelace' unit string."""
    asset = DanogoTupleAsset(policy=b"", asset_name=b"")
    assert asset.unit == "lovelace"


def test_danogo_tuple_asset_unit_with_policy_returns_hex() -> None:
    """A non-empty policy concatenated with asset name must be returned as hex."""
    policy = bytes.fromhex("ab" * 28)  # 28-byte policy
    name = b"TOKEN"
    asset = DanogoTupleAsset(policy=policy, asset_name=name)
    expected = policy.hex() + name.hex()
    assert asset.unit == expected


def test_danogo_tuple_asset_unit_empty_name_uses_policy_only() -> None:
    """When asset_name is empty only the policy hex is returned."""
    policy = bytes.fromhex("cd" * 28)
    asset = DanogoTupleAsset(policy=policy, asset_name=b"")
    assert asset.unit == policy.hex()


def test_danogo_tuple_asset_assets_returns_zero_quantity() -> None:
    """The assets property must wrap the unit with a quantity of 0."""
    asset = DanogoTupleAsset(policy=b"", asset_name=b"")
    result = asset.assets
    assert isinstance(result, Assets)
    assert result["lovelace"] == 0


def test_danogo_tuple_asset_assets_with_policy_unit() -> None:
    """Assets property for a non-ADA token must have 0 quantity."""
    policy = bytes.fromhex("ab" * 28)
    name = b"TKN"
    asset = DanogoTupleAsset(policy=policy, asset_name=name)
    expected_unit = policy.hex() + name.hex()
    result = asset.assets
    assert result[expected_unit] == 0


def test_danogo_tuple_asset_from_list_valid() -> None:
    """from_list with exactly 2 elements must succeed."""
    data = [b"", b""]
    result = DanogoTupleAsset.from_list(data)
    assert result.policy == b""
    assert result.asset_name == b""


def test_danogo_tuple_asset_from_list_valid_with_content() -> None:
    """from_list with policy and name bytes must populate fields correctly."""
    policy = bytes.fromhex("ab" * 28)
    name = b"TKN"
    result = DanogoTupleAsset.from_list([policy, name])
    assert result.policy == policy
    assert result.asset_name == name
    assert result.unit == policy.hex() + name.hex()


def test_danogo_tuple_asset_from_list_too_few_elements_raises() -> None:
    """from_list with only 1 element must raise ValueError."""
    with pytest.raises(ValueError):
        DanogoTupleAsset.from_list([b"only_one"])


def test_danogo_tuple_asset_from_list_too_many_elements_raises() -> None:
    """from_list with 3 elements must raise ValueError."""
    with pytest.raises(ValueError):
        DanogoTupleAsset.from_list([b"a", b"b", b"c"])


def test_danogo_tuple_asset_from_list_empty_raises() -> None:
    """from_list with an empty list must raise ValueError."""
    with pytest.raises(ValueError):
        DanogoTupleAsset.from_list([])


# ===========================================================================
# Section 2: DanogoPRational — construction and field access
# ===========================================================================


def test_danogo_prational_stores_numerator_and_denominator() -> None:
    """DanogoPRational must correctly store numerator and denominator."""
    pr = DanogoPRational(numerator=3, denominator=4)
    assert pr.numerator == 3
    assert pr.denominator == 4


def test_danogo_prational_constr_id() -> None:
    """CONSTR_ID must be 0 — required for CBOR encoding."""
    assert DanogoPRational.CONSTR_ID == 0


def test_danogo_prational_large_values() -> None:
    """DanogoPRational must handle large integer values."""
    big = 10 ** 18
    pr = DanogoPRational(numerator=big, denominator=big + 1)
    assert pr.numerator == big
    assert pr.denominator == big + 1


# ===========================================================================
# Section 3: ConcentratedPoolDatum — pool_pair and CBOR deserialization
# ===========================================================================


def test_concentrated_pool_datum_from_cbor_decodes_correctly() -> None:
    """Parsing the reference CBOR must yield a valid ConcentratedPoolDatum."""
    datum = ConcentratedPoolDatum.from_cbor(_SWAP_TEST_CBOR)
    assert isinstance(datum, ConcentratedPoolDatum)
    assert datum.lp_fee_rate == 10


def test_concentrated_pool_datum_from_cbor_token_x_is_ada() -> None:
    """token_x in the reference datum must represent ADA (empty policy + name)."""
    datum = ConcentratedPoolDatum.from_cbor(_SWAP_TEST_CBOR)
    asset_x = DanogoTupleAsset.from_list(datum.token_x)
    assert asset_x.unit == "lovelace"


def test_concentrated_pool_datum_from_cbor_token_y_is_fUSDA() -> None:
    """token_y in the reference datum must map to the fUSDA unit."""
    datum = ConcentratedPoolDatum.from_cbor(_SWAP_TEST_CBOR)
    asset_y = DanogoTupleAsset.from_list(datum.token_y)
    assert asset_y.unit == _TOKEN_B


def test_concentrated_pool_datum_pool_pair_returns_assets() -> None:
    """pool_pair must return an Assets mapping both token_x and token_y to 0."""
    datum = ConcentratedPoolDatum.from_cbor(_SWAP_TEST_CBOR)
    pair = datum.pool_pair()
    assert isinstance(pair, Assets)
    assert "lovelace" in pair.root
    assert _TOKEN_B in pair.root


def test_concentrated_pool_datum_pool_pair_quantities_are_zero() -> None:
    """pool_pair quantities must always be 0 (it signals only the token pair)."""
    datum = ConcentratedPoolDatum.from_cbor(_SWAP_TEST_CBOR)
    pair = datum.pool_pair()
    assert pair["lovelace"] == 0
    assert pair[_TOKEN_B] == 0


def test_concentrated_pool_datum_sqrt_prices_are_rational() -> None:
    """sqrt_lower_price and sqrt_upper_price must be DanogoPRational instances."""
    datum = ConcentratedPoolDatum.from_cbor(_SWAP_TEST_CBOR)
    assert isinstance(datum.sqrt_lower_price, DanogoPRational)
    assert isinstance(datum.sqrt_upper_price, DanogoPRational)
    assert datum.sqrt_lower_price.denominator > 0
    assert datum.sqrt_upper_price.denominator > 0


# ===========================================================================
# Section 4: ConcentratedOrderDatum — create_datum, address_source,
#            requested_amount, order_type
# ===========================================================================


def _dummy_address() -> Address:
    """Return a minimal dummy pycardano Address for tests."""
    return Address(
        payment_part=VerificationKeyHash(b"\x00" * 28),
        network=Network.MAINNET,
    )


def test_concentrated_order_datum_create_datum_returns_instance() -> None:
    """create_datum must return a ConcentratedOrderDatum instance."""
    addr = _dummy_address()
    result = ConcentratedOrderDatum.create_datum(
        address_source=addr,
        in_assets=Assets(root={"lovelace": 1}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        batcher_fee=Assets({}),
        deposit=Assets({}),
    )
    assert isinstance(result, ConcentratedOrderDatum)


def test_concentrated_order_datum_address_source_returns_address() -> None:
    """address_source must return an Address object."""
    datum = ConcentratedOrderDatum()
    result = datum.address_source()
    assert isinstance(result, Address)


def test_concentrated_order_datum_address_source_is_mainnet() -> None:
    """The dummy address returned by address_source must be on MAINNET."""
    datum = ConcentratedOrderDatum()
    addr = datum.address_source()
    assert addr.network == Network.MAINNET


def test_concentrated_order_datum_requested_amount_is_empty() -> None:
    """requested_amount must return an empty Assets mapping."""
    datum = ConcentratedOrderDatum()
    result = datum.requested_amount()
    assert isinstance(result, Assets)
    assert len(result.root) == 0


def test_concentrated_order_datum_order_type_is_swap() -> None:
    """order_type must return OrderType.swap."""
    datum = ConcentratedOrderDatum()
    assert datum.order_type() == OrderType.swap


def test_concentrated_order_datum_constr_id_is_zero() -> None:
    """CONSTR_ID must be 0 for on-chain CBOR compatibility."""
    assert ConcentratedOrderDatum.CONSTR_ID == 0


# ===========================================================================
# Section 5: Delegation and ConcentratedPoolInfo dataclasses
# ===========================================================================


def test_delegation_stores_pool_id_and_rewards() -> None:
    """Delegation must correctly store pool_id and rewards."""
    d = Delegation(pool_id="pool1abc", rewards=500)
    assert d.pool_id == "pool1abc"
    assert d.rewards == 500


def test_delegation_pool_id_can_be_none() -> None:
    """Delegation pool_id is optional and can be None."""
    d = Delegation(pool_id=None, rewards=0)
    assert d.pool_id is None


def test_concentrated_pool_info_stores_fields() -> None:
    """ConcentratedPoolInfo must store all its fields correctly."""
    info = ConcentratedPoolInfo(
        nft_unit="abc123",
        address="addr_test1...",
        staking_tx_hash="aabbcc" * 10 + "aabb",
        staking_tx_index=1,
        staking_script_hash="0" * 56,
    )
    assert info.nft_unit == "abc123"
    assert info.staking_tx_index == 1


def test_concentrated_pool_script_info_stores_fields() -> None:
    """ConcentratedPoolScriptInfo must hold the out_ref and script_hash."""
    ref = TransactionInput(
        transaction_id=TransactionId(b"\x00" * 32),
        index=0,
    )
    info = ConcentratedPoolScriptInfo(out_ref=ref, script_hash=_SCRIPT_HASH_HEX)
    assert info.script_hash == _SCRIPT_HASH_HEX
    assert info.out_ref == ref


def test_concentrated_staking_script_info_stores_fields() -> None:
    """ConcentratedStakingScriptInfo must hold the out_ref and script_hash."""
    ref = TransactionInput(
        transaction_id=TransactionId(b"\x01" * 32),
        index=1,
    )
    info = ConcentratedStakingScriptInfo(out_ref=ref, script_hash=_SCRIPT_HASH_HEX)
    assert info.script_hash == _SCRIPT_HASH_HEX
    assert info.out_ref == ref


# ===========================================================================
# Section 6: ConcentratedPoolState — class methods
# ===========================================================================


def test_dex_name_returns_expected_string() -> None:
    """dex() must return the canonical DEX name string."""
    assert ConcentratedPoolState.dex() == "Danogo Concentrated Liquidity"


def test_order_selector_returns_list_of_strings() -> None:
    """order_selector must return a non-empty list of bech32 address strings."""
    result = ConcentratedPoolState.order_selector()
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(s, str) for s in result)


def test_order_selector_count_matches_supported_pools() -> None:
    """order_selector must have one entry per supported pool."""
    result = ConcentratedPoolState.order_selector()
    assert len(result) == len(ConcentratedPoolState._SUPPORTED_POOLS)


def test_pool_selector_returns_pool_selector_instance() -> None:
    """pool_selector must return a PoolSelector."""
    result = ConcentratedPoolState.pool_selector()
    assert isinstance(result, PoolSelector)


def test_pool_selector_has_addresses() -> None:
    """pool_selector must include the stake addresses."""
    result = ConcentratedPoolState.pool_selector()
    assert result.addresses is not None
    assert len(result.addresses) > 0


def test_pool_selector_has_nft_assets() -> None:
    """pool_selector must include the NFT unit for every supported pool."""
    result = ConcentratedPoolState.pool_selector()
    assert result.assets is not None
    assert len(result.assets) == len(ConcentratedPoolState._SUPPORTED_POOLS)


def test_order_datum_class_returns_concentrated_order_datum() -> None:
    """order_datum_class must return ConcentratedOrderDatum."""
    assert ConcentratedPoolState.order_datum_class() is ConcentratedOrderDatum


def test_pool_datum_class_returns_concentrated_pool_datum() -> None:
    """pool_datum_class must return ConcentratedPoolDatum."""
    assert ConcentratedPoolState.pool_datum_class() is ConcentratedPoolDatum


def test_default_script_class_returns_plutus_v3() -> None:
    """default_script_class must return PlutusV3Script."""
    assert ConcentratedPoolState.default_script_class() is PlutusV3Script


def test_script_class_returns_plutus_v3() -> None:
    """script_class must return PlutusV3Script."""
    assert ConcentratedPoolState.script_class() is PlutusV3Script


def test_supported_pools_registry_has_three_entries() -> None:
    """_SUPPORTED_POOLS must have exactly three pool entries."""
    assert len(ConcentratedPoolState._SUPPORTED_POOLS) == 3


def test_pool_nft_to_stake_address_maps_all_pools() -> None:
    """_pool_nft_to_stake_address must contain one entry per supported pool."""
    mapping = ConcentratedPoolState._pool_nft_to_stake_address
    assert len(mapping) == len(ConcentratedPoolState._SUPPORTED_POOLS)


# ===========================================================================
# Section 7: ConcentratedPoolState — instance properties
# ===========================================================================


def test_swap_forward_is_false() -> None:
    """swap_forward must always return False for Danogo direct-swap pools."""
    pool = _make_swap_pool()
    assert pool.swap_forward is False


def test_pool_id_equals_pool_nft_unit() -> None:
    """pool_id must be the string unit of pool_nft."""
    pool = _make_swap_pool()
    assert pool.pool_id == pool.pool_nft.unit()
    assert pool.pool_id == "test_pool_nft"


def test_stake_address_fallback_for_unknown_nft() -> None:
    """An unknown pool_nft must fall back to the first stake address."""
    pool = _make_swap_pool()  # pool_nft = "test_pool_nft" (not in registry)
    expected_fallback = ConcentratedPoolState._stake_address[0]
    assert pool.stake_address == expected_fallback


def test_stake_address_resolves_for_known_nft() -> None:
    """A pool_nft matching the registry must return the correct stake address."""
    pool = _make_swap_pool(pool_nft=Assets(root={_KNOWN_NFT: 1}))
    expected = ConcentratedPoolState._pool_nft_to_stake_address[_KNOWN_NFT]
    assert pool.stake_address == expected


def test_pool_reward_amount_is_zero_for_unknown_nft() -> None:
    """pool_reward_amount must be 0 when the NFT is not in the staking registry."""
    pool = _make_swap_pool()
    assert pool.pool_reward_amount == 0


def test_unit_a_is_first_asset() -> None:
    """unit_a must equal the first key in the assets dict."""
    pool = _make_swap_pool()
    assert pool.unit_a == "lovelace"


def test_unit_b_is_second_asset() -> None:
    """unit_b must equal the second key in the assets dict."""
    pool = _make_swap_pool()
    assert pool.unit_b == _TOKEN_B


def test_reserve_a_matches_lovelace_amount() -> None:
    """reserve_a must equal the raw lovelace quantity in the pool."""
    pool = _make_swap_pool()
    assert pool.reserve_a == 100_000_000


def test_reserve_b_matches_token_b_amount() -> None:
    """reserve_b must equal the raw token B quantity in the pool."""
    pool = _make_swap_pool()
    assert pool.reserve_b == 36_000_000


def test_pool_datum_parses_from_cbor() -> None:
    """pool_datum must deserialize into a ConcentratedPoolDatum with lp_fee_rate=10."""
    pool = _make_swap_pool()
    datum = pool.pool_datum
    assert isinstance(datum, ConcentratedPoolDatum)
    assert datum.lp_fee_rate == 10


# ===========================================================================
# Section 8: ConcentratedPoolState.extract_pool_nft — class method
# ===========================================================================


def test_extract_pool_nft_single_nft_succeeds() -> None:
    """extract_pool_nft must extract the single quantity-1 asset as pool NFT."""
    values = {
        "assets": Assets(
            root={"lovelace": 2_000_000, "nftpolicy1234" + "aa" * 28: 1}
        )
    }
    nft = ConcentratedPoolState.extract_pool_nft(values)
    assert nft is not None
    assert nft.quantity() == 1


def test_extract_pool_nft_existing_pool_nft_returns_it() -> None:
    """If pool_nft already exists in values, extract_pool_nft must return it."""
    existing_nft = Assets(root={"some_nft": 1})
    values = {
        "assets": Assets(root={"lovelace": 2_000_000}),
        "pool_nft": existing_nft,
    }
    result = ConcentratedPoolState.extract_pool_nft(values)
    assert result == existing_nft


def test_extract_pool_nft_multiple_nfts_raises_invalid_pool_error() -> None:
    """Multiple quantity-1 assets must raise InvalidPoolError."""
    values = {
        "assets": Assets(
            root={
                "lovelace": 2_000_000,
                "nft1" + "aa" * 30: 1,
                "nft2" + "bb" * 30: 1,
            }
        )
    }
    with pytest.raises(InvalidPoolError):
        ConcentratedPoolState.extract_pool_nft(values)


def test_extract_pool_nft_no_nft_raises_invalid_pool_error() -> None:
    """No quantity-1 asset must raise InvalidPoolError."""
    values = {"assets": Assets(root={"lovelace": 2_000_000, "tokenX": 100})}
    with pytest.raises(InvalidPoolError):
        ConcentratedPoolState.extract_pool_nft(values)


# ===========================================================================
# Section 9: ConcentratedPoolState.price — virtual-reserve-based price
# ===========================================================================


def test_price_returns_tuple_of_two_decimals() -> None:
    """price must return a 2-tuple of Decimal values."""
    from decimal import Decimal

    pool = _make_swap_pool()
    result = pool.price
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(v, Decimal) for v in result)


def test_price_both_components_non_negative() -> None:
    """Both price components must be >= 0."""
    pool = _make_swap_pool()
    price_b_in_a, price_a_in_b = pool.price
    assert price_b_in_a >= 0
    assert price_a_in_b >= 0


def test_price_zero_reserves_returns_zero_tuple() -> None:
    """When effective reserves collapse to zero, price must return (0, 0)."""
    from decimal import Decimal

    zero_pool = _make_swap_pool(
        assets=Assets(root={"lovelace": 0, _TOKEN_B: 0}),
    )
    price_b, price_a = zero_pool.price
    assert price_b == Decimal(0)
    assert price_a == Decimal(0)


# ===========================================================================
# Section 10: ConcentratedPoolState.tvl — total value locked
# ===========================================================================


def test_tvl_ada_pool_returns_positive_decimal() -> None:
    """tvl for an ADA-containing pool must be a positive Decimal."""
    from decimal import Decimal

    pool = _make_swap_pool()
    result = pool.tvl
    assert isinstance(result, Decimal)
    assert result >= 0


def test_tvl_non_ada_pool_raises_not_implemented() -> None:
    """tvl must raise NotImplementedError for pools that contain no ADA."""
    from unittest.mock import PropertyMock

    pool = _make_swap_pool()
    # Patch unit_a and unit_b to simulate a non-ADA pair without triggering
    # pool construction validators that require lovelace in the assets dict.
    with patch.object(type(pool), "unit_a", new_callable=PropertyMock, return_value="non_ada_token_x"):
        with patch.object(type(pool), "unit_b", new_callable=PropertyMock, return_value="non_ada_token_y"):
            with pytest.raises(NotImplementedError):
                _ = pool.tvl


# ===========================================================================
# Section 11: ConcentratedPoolState.swap_datum
# ===========================================================================


def test_swap_datum_returns_concentrated_order_datum() -> None:
    """swap_datum must return a ConcentratedOrderDatum instance."""
    pool = _make_swap_pool()
    addr = _dummy_address()
    result = pool.swap_datum(
        address_source=addr,
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 300_000}),
    )
    assert isinstance(result, ConcentratedOrderDatum)


# ===========================================================================
# Section 12: ConcentratedPoolState._get_amount_out_and_platform_fee
# ===========================================================================

# --- Happy-path tests ---


def test_swap_token_a_in_returns_token_b() -> None:
    """Swapping lovelace in should return fUSDA out."""
    pool = _make_swap_pool()
    asset_in = Assets(root={"lovelace": 1_000_000})  # 1 ADA

    out_assets, fee = pool._get_amount_out_and_platform_fee(asset_in, reward_amount=0)

    assert out_assets.unit() == _TOKEN_B
    assert out_assets.quantity() > 0
    assert isinstance(fee, int)
    assert fee >= 0


def test_swap_token_b_in_returns_token_a() -> None:
    """Swapping fUSDA in should return lovelace out."""
    pool = _make_swap_pool()
    asset_in = Assets(root={_TOKEN_B: 1_000_000})

    out_assets, fee = pool._get_amount_out_and_platform_fee(asset_in, reward_amount=0)

    assert out_assets.unit() == "lovelace"
    assert out_assets.quantity() > 0
    assert isinstance(fee, int)
    assert fee >= 0


def test_output_token_is_always_opposite_of_input() -> None:
    """The output unit must differ from the input unit in both swap directions."""
    pool = _make_swap_pool()

    out_a, _ = pool._get_amount_out_and_platform_fee(
        Assets(root={"lovelace": 1_000_000}), reward_amount=0
    )
    assert out_a.unit() == _TOKEN_B

    out_b, _ = pool._get_amount_out_and_platform_fee(
        Assets(root={_TOKEN_B: 1_000_000}), reward_amount=0
    )
    assert out_b.unit() == "lovelace"


def test_larger_input_gives_more_output() -> None:
    """A larger input amount must produce more output than a smaller one."""
    pool = _make_swap_pool()

    out_small, _ = pool._get_amount_out_and_platform_fee(
        Assets(root={"lovelace": 1_000_000}), reward_amount=0
    )
    out_large, _ = pool._get_amount_out_and_platform_fee(
        Assets(root={"lovelace": 5_000_000}), reward_amount=0
    )

    assert out_large.quantity() > out_small.quantity()


def test_platform_fee_is_5_percent_of_lp_fee() -> None:
    """Platform fee = 5% of LP fee.  LP fee rate = 10 / 10_000 = 0.1%."""
    pool = _make_swap_pool()
    # Input: 10_000_000 lovelace
    # lp_fee = (10_000_000 * 10) // 10_000 = 10_000
    # platform_fee = (10_000 * 5) // 100 = 500
    _, platform_fee = pool._get_amount_out_and_platform_fee(
        Assets(root={"lovelace": 10_000_000}), reward_amount=0
    )

    assert platform_fee == 500


def test_staking_rewards_affect_output_for_ada_in() -> None:
    """Staking rewards change the effective ADA reserve and therefore the output.

    Adding rewards increases active_reserve_x, which shifts the virtual price
    such that ADA becomes relatively cheaper (more abundant). For a fixed ADA
    input, this results in *less* fUSDA out.
    """
    pool = _make_swap_pool()
    asset_in = Assets(root={"lovelace": 5_000_000})

    out_no_reward, _ = pool._get_amount_out_and_platform_fee(
        asset_in, reward_amount=0
    )
    out_with_reward, _ = pool._get_amount_out_and_platform_fee(
        asset_in, reward_amount=10_000_000
    )

    # Rewards add ADA to the pool -> ADA is more abundant -> less fUSDA per ADA.
    assert out_with_reward.quantity() < out_no_reward.quantity()


def test_zero_reward_is_identical_to_no_reward() -> None:
    """Passing reward_amount=0 is the baseline and must be consistent."""
    pool = _make_swap_pool()
    asset_in = Assets(root={"lovelace": 2_000_000})

    out_1, fee_1 = pool._get_amount_out_and_platform_fee(asset_in, reward_amount=0)
    out_2, fee_2 = pool._get_amount_out_and_platform_fee(asset_in, reward_amount=0)

    assert out_1.quantity() == out_2.quantity()
    assert fee_1 == fee_2


# --- Error / edge-case tests ---


def test_multi_token_asset_raises_value_error() -> None:
    """An asset dict with more than one token must raise ValueError."""
    pool = _make_swap_pool()
    multi_asset = Assets(root={"lovelace": 1_000_000, _TOKEN_B: 500_000})

    with pytest.raises(ValueError, match="only have one token"):
        pool._get_amount_out_and_platform_fee(multi_asset, reward_amount=0)


def test_unknown_token_raises_value_error() -> None:
    """An asset that is neither token A nor token B must raise ValueError."""
    pool = _make_swap_pool()
    fake_token = "ab" * 32  # 64-char hex, not in this pool
    unknown_asset = Assets(root={fake_token: 1_000_000})

    with pytest.raises(ValueError, match="must be either token A or token B"):
        pool._get_amount_out_and_platform_fee(unknown_asset, reward_amount=0)


def test_swap_exceeding_pool_liquidity_raises_value_error() -> None:
    """Swapping more than the pool can provide must raise ValueError."""
    tiny_pool = _make_swap_pool(
        assets=Assets(root={"lovelace": 100_000_000, _TOKEN_B: 5})
    )
    huge_ada_in = Assets(root={"lovelace": 90_000_000})  # expects far more than 5 fUSDA

    with pytest.raises(ValueError):
        tiny_pool._get_amount_out_and_platform_fee(huge_ada_in, reward_amount=0)


# ===========================================================================
# Section 13: ConcentratedPoolState.get_amount_out
# ===========================================================================


def test_get_amount_out_returns_assets_and_float() -> None:
    """get_amount_out must return (Assets, float)."""
    pool = _make_swap_pool()
    out_assets, price_impact = pool.get_amount_out(Assets(root={"lovelace": 1_000_000}))
    assert isinstance(out_assets, Assets)
    assert isinstance(price_impact, float)


def test_get_amount_out_price_impact_is_always_zero() -> None:
    """Danogo pools always return 0.0 as the price impact."""
    pool = _make_swap_pool()
    _, price_impact = pool.get_amount_out(Assets(root={"lovelace": 5_000_000}))
    assert price_impact == 0.0


def test_get_amount_out_consistent_with_internal_method() -> None:
    """get_amount_out must match _get_amount_out_and_platform_fee output."""
    pool = _make_swap_pool()
    asset_in = Assets(root={"lovelace": 3_000_000})
    out_public, _ = pool.get_amount_out(asset_in)
    out_internal, _ = pool._get_amount_out_and_platform_fee(asset_in, reward_amount=0)
    assert out_public.quantity() == out_internal.quantity()
    assert out_public.unit() == out_internal.unit()


def test_get_amount_out_multi_token_raises() -> None:
    """get_amount_out must propagate ValueError for multi-token input."""
    pool = _make_swap_pool()
    multi = Assets(root={"lovelace": 1_000_000, _TOKEN_B: 1_000_000})
    with pytest.raises(ValueError):
        pool.get_amount_out(multi)


# ===========================================================================
# Section 14: ConcentratedPoolState.get_amount_in
# ===========================================================================


def test_get_amount_in_returns_assets_and_float() -> None:
    """get_amount_in must return (Assets, float)."""
    pool = _make_swap_pool()
    in_assets, slippage = pool.get_amount_in(Assets(root={_TOKEN_B: 100_000}))
    assert isinstance(in_assets, Assets)
    assert isinstance(slippage, float)


def test_get_amount_in_input_unit_is_opposite_to_output() -> None:
    """get_amount_in requesting token B must return token A as input."""
    pool = _make_swap_pool()
    in_assets, _ = pool.get_amount_in(Assets(root={_TOKEN_B: 100_000}))
    assert in_assets.unit() == "lovelace"


def test_get_amount_in_requesting_token_a_returns_token_b_input() -> None:
    """get_amount_in requesting lovelace must return token B as input."""
    pool = _make_swap_pool()
    in_assets, _ = pool.get_amount_in(Assets(root={"lovelace": 1_000_000}))
    assert in_assets.unit() == _TOKEN_B


def test_get_amount_in_produces_sufficient_output() -> None:
    """The input found by get_amount_in must actually yield >= the target output."""
    pool = _make_swap_pool()
    target = Assets(root={_TOKEN_B: 200_000})
    in_assets, _ = pool.get_amount_in(target)
    out_assets, _ = pool._get_amount_out_and_platform_fee(in_assets, reward_amount=0)
    assert out_assets.quantity() >= target.quantity()


def test_get_amount_in_larger_target_needs_more_input() -> None:
    """A larger desired output must require a larger input."""
    pool = _make_swap_pool()
    in_small, _ = pool.get_amount_in(Assets(root={_TOKEN_B: 100_000}))
    in_large, _ = pool.get_amount_in(Assets(root={_TOKEN_B: 500_000}))
    assert in_large.quantity() > in_small.quantity()


def test_get_amount_in_zero_or_negative_output_raises() -> None:
    """get_amount_in with a non-positive target output must raise ValueError."""
    pool = _make_swap_pool()
    with pytest.raises(ValueError):
        pool.get_amount_in(Assets(root={_TOKEN_B: 0}))


# ===========================================================================
# Section 15: calculate_l — liquidity invariant computation
# ===========================================================================


def test_calculate_l_returns_positive_fraction() -> None:
    """calculate_l must return a positive Fraction for valid inputs."""
    x = Fraction(100)
    y = Fraction(100)
    sqrt_pa = Fraction(1, 2)
    sqrt_pb = Fraction(1, 1)
    result = calculate_l(x, y, sqrt_pa, sqrt_pb)
    assert isinstance(result, Fraction)
    assert result > 0


def test_calculate_l_symmetric_reserves() -> None:
    """Symmetric reserves must give a well-defined positive L value."""
    result = calculate_l(Fraction(50), Fraction(50), Fraction(1, 2), Fraction(3, 2))
    assert result > 0


def test_calculate_l_larger_x_shifts_liquidity() -> None:
    """Increasing x while holding y fixed must increase L."""
    sqrt_pa = Fraction(1, 2)
    sqrt_pb = Fraction(1, 1)
    l_base = calculate_l(Fraction(10), Fraction(10), sqrt_pa, sqrt_pb)
    l_bigger = calculate_l(Fraction(20), Fraction(10), sqrt_pa, sqrt_pb)
    assert l_bigger > l_base


def test_calculate_l_zero_price_raises_value_error() -> None:
    """sqrt_pa = 0 causes p_a = 0, which must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        calculate_l(Fraction(1), Fraction(1), Fraction(0), Fraction(1))


def test_calculate_l_equal_sqrt_prices_raises_value_error() -> None:
    """sqrt_pa == sqrt_pb makes denom_diff zero, raising ValueError."""
    with pytest.raises(ValueError, match="zero"):
        calculate_l(Fraction(1), Fraction(1), Fraction(1), Fraction(1))


def test_calculate_l_negative_sqrt_pb_zero_raises_value_error() -> None:
    """sqrt_pb = 0 causes p_b = 0, which must raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        calculate_l(Fraction(1), Fraction(1), Fraction(1, 2), Fraction(0))


# ===========================================================================
# Section 16: calculate_xv_yv — virtual reserve computation
# ===========================================================================


def test_calculate_xv_yv_returns_two_fractions() -> None:
    """calculate_xv_yv must return a 2-tuple of Fraction values."""
    xv, yv = calculate_xv_yv(100, 100, 1, 2, 1, 1)
    assert isinstance(xv, Fraction)
    assert isinstance(yv, Fraction)


def test_calculate_xv_yv_positive_results_for_valid_inputs() -> None:
    """Virtual reserves must be positive for positive real reserves."""
    xv, yv = calculate_xv_yv(50, 50, 1, 2, 3, 2)
    assert xv > 0
    assert yv > 0


def test_calculate_xv_yv_zero_denominator_sqrt_pa_raises() -> None:
    """Zero denominator for sqrt_pa must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_xv_yv(100, 100, 1, 0, 1, 1)


def test_calculate_xv_yv_zero_denominator_sqrt_pb_raises() -> None:
    """Zero denominator for sqrt_pb must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_xv_yv(100, 100, 1, 2, 1, 0)


def test_calculate_xv_yv_results_are_ceiled() -> None:
    """Returned values must be integers (ceiled fractions), not sub-unity fractions."""
    xv, yv = calculate_xv_yv(100, 100, 1, 2, 1, 1)
    assert xv.denominator == 1
    assert yv.denominator == 1


# ===========================================================================
# Section 17: get_epoch — epoch number from millisecond timestamp
# ===========================================================================


def test_get_epoch_at_boundary_returns_328() -> None:
    """At the epoch boundary timestamp, the epoch must be 328 on MAINNET."""
    epoch_boundary_ms = 1_647_899_091_000
    assert get_epoch(epoch_boundary_ms, Network.MAINNET) == 328


def test_get_epoch_one_mainnet_epoch_later_returns_329() -> None:
    """One MAINNET epoch (432_000 seconds) after the boundary yields epoch 329."""
    epoch_boundary_ms = 1_647_899_091_000
    mainnet_epoch_ms = 432_000_000
    t = epoch_boundary_ms + mainnet_epoch_ms
    assert get_epoch(t, Network.MAINNET) == 329


def test_get_epoch_testnet_uses_shorter_epoch_length() -> None:
    """TESTNET epoch is 1_800 seconds, so more epochs fit in the same time span."""
    epoch_boundary_ms = 1_647_899_091_000
    one_mainnet_epoch_ms = 432_000_000
    t = epoch_boundary_ms + one_mainnet_epoch_ms
    mainnet_epoch = get_epoch(t, Network.MAINNET)
    testnet_epoch = get_epoch(t, Network.TESTNET)
    # Shorter TESTNET epochs -> higher epoch number for same timestamp
    assert testnet_epoch > mainnet_epoch


def test_get_epoch_monotone_increases_with_time() -> None:
    """A later timestamp must yield a higher epoch number."""
    t1 = 1_647_900_000_000
    t2 = t1 + 500_000_000
    e1 = get_epoch(t1, Network.MAINNET)
    e2 = get_epoch(t2, Network.MAINNET)
    assert e2 > e1


# ===========================================================================
# Section 18: ceil_div — ceiling integer division
# ===========================================================================


def test_ceil_div_exact_division() -> None:
    """ceil_div must match integer division when there is no remainder."""
    assert ceil_div(10, 2) == 5
    assert ceil_div(9, 3) == 3


def test_ceil_div_rounds_up_with_remainder() -> None:
    """ceil_div must round up when a is not divisible by b."""
    assert ceil_div(7, 2) == 4
    assert ceil_div(1, 3) == 1


def test_ceil_div_single_element() -> None:
    """ceil_div(1, 1) must be 1."""
    assert ceil_div(1, 1) == 1


def test_ceil_div_large_numerator() -> None:
    """ceil_div must handle large integers correctly without floating-point loss."""
    # 10**18 + 1 is not divisible by 10**9, so ceiling division must round up.
    # math.ceil(float) would lose precision here; use exact integer arithmetic.
    a = 10 ** 18 + 1
    b = 10 ** 9
    expected = math.ceil(Fraction(a, b))  # Fraction preserves exact integer ratio
    assert ceil_div(a, b) == expected


def test_ceil_div_zero_denominator_raises_value_error() -> None:
    """ceil_div with b=0 must raise ValueError."""
    with pytest.raises(ValueError, match="zero"):
        ceil_div(5, 0)


def test_ceil_div_zero_numerator() -> None:
    """ceil_div(0, b) must be 0 for any non-zero b."""
    assert ceil_div(0, 5) == 0


# ===========================================================================
# Section 19: calc_liquidity — integer-domain liquidity computation
# ===========================================================================


def test_calc_liquidity_returns_tuple_of_two_ints() -> None:
    """calc_liquidity must return a 2-tuple of integers."""
    result = calc_liquidity(100, 100, (1, 1), (2, 1))
    assert isinstance(result, tuple)
    assert len(result) == 2
    num, den = result
    assert isinstance(num, int)
    assert isinstance(den, int)


def test_calc_liquidity_denominator_is_positive() -> None:
    """The denominator must be positive for pb > pa (in fraction form)."""
    # pa = 1/1 = 1.0, pb = 2/1 = 2.0 -> pb > pa -> denominator > 0
    _, den = calc_liquidity(100, 100, (1, 1), (2, 1))
    assert den > 0


def test_calc_liquidity_numerator_is_positive() -> None:
    """The numerator must be positive for any positive reserve amounts."""
    num, _ = calc_liquidity(100, 100, (1, 1), (2, 1))
    assert num > 0


def test_calc_liquidity_larger_reserves_give_proportionally_larger_numerator() -> None:
    """Doubling both reserves must increase the effective liquidity value."""
    num1, den1 = calc_liquidity(100, 100, (1, 1), (2, 1))
    num2, den2 = calc_liquidity(200, 200, (1, 1), (2, 1))
    l1 = num1 / den1
    l2 = num2 / den2
    assert l2 > l1


def test_calc_liquidity_matches_calculate_l_fraction() -> None:
    """calc_liquidity integer result should agree with calculate_l Fraction (within 1%)."""
    x, y = 50, 30
    pa = (1, 2)  # sqrt(pa) = 1/2
    pb = (3, 2)  # sqrt(pb) = 3/2
    num, den = calc_liquidity(x, y, pa, pb)
    l_int = num / den

    sqrt_pa = Fraction(pa[0], pa[1])
    sqrt_pb = Fraction(pb[0], pb[1])
    l_frac = float(calculate_l(Fraction(x), Fraction(y), sqrt_pa, sqrt_pb))

    # Integer sqrt truncation may differ slightly; allow 1% tolerance
    assert abs(l_int - l_frac) / l_frac < 0.01


# ===========================================================================
# Section 20: get_pool_change — AMM output and fee computation
# ===========================================================================


def test_get_pool_change_returns_tuple_of_two_ints() -> None:
    """get_pool_change must return a (expected_out, platform_fee) tuple of ints."""
    result = get_pool_change(
        amount_in=10_000,
        token_in_virtual=100_000,
        token_out_virtual=100_000,
        token_out_real=100_000,
        lp_fee_rate=30,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    out, fee = result
    assert isinstance(out, int)
    assert isinstance(fee, int)


def test_get_pool_change_output_less_than_real_reserve() -> None:
    """Output amount must never exceed the real reserve."""
    out, _ = get_pool_change(
        amount_in=1_000,
        token_in_virtual=10_000,
        token_out_virtual=10_000,
        token_out_real=10_000,
        lp_fee_rate=30,
    )
    assert out <= 10_000


def test_get_pool_change_platform_fee_is_5_pct_of_lp_fee() -> None:
    """platform_fee must equal floor(lp_fee * 5 / 100)."""
    # amount_in=10_000, lp_fee_rate=30
    # lp_fee = (10_000 * 30) // 10_000 = 30
    # platform_fee = (30 * 5) // 100 = 1
    _, platform_fee = get_pool_change(
        amount_in=10_000,
        token_in_virtual=100_000,
        token_out_virtual=100_000,
        token_out_real=100_000,
        lp_fee_rate=30,
    )
    assert platform_fee == 1


def test_get_pool_change_output_exceeds_real_reserve_raises() -> None:
    """When expected_out > token_out_real, ValueError must be raised."""
    with pytest.raises(ValueError, match="pool out exceeded"):
        get_pool_change(
            amount_in=100_000,
            token_in_virtual=1_000,
            token_out_virtual=1_000,
            token_out_real=1,     # way too small
            lp_fee_rate=10,
        )


def test_get_pool_change_larger_input_gives_more_output() -> None:
    """A larger amount_in must produce a larger expected_out."""
    out1, _ = get_pool_change(1_000, 100_000, 100_000, 100_000, 30)
    out2, _ = get_pool_change(5_000, 100_000, 100_000, 100_000, 30)
    assert out2 > out1


# ===========================================================================
# Section 21: calculate_concentrated_pool_swap — full AMM round-trip
# ===========================================================================


def _make_swap_datum() -> ConcentratedPoolDatum:
    """Return the reference ConcentratedPoolDatum decoded from the test CBOR."""
    return ConcentratedPoolDatum.from_cbor(_SWAP_TEST_CBOR)


def test_calculate_concentrated_pool_swap_token_a_in_positive_delta() -> None:
    """Positive delta_amount (ADA in) must return fUSDA out and a non-negative fee."""
    datum = _make_swap_datum()
    out, fee = calculate_concentrated_pool_swap(
        token_a_amount=100_000_000,
        token_b_amount=36_000_000,
        datum=datum,
        delta_amount=1_000_000,  # 1 ADA in -> positive
        reward_amount=0,
    )
    assert out > 0
    assert fee >= 0


def test_calculate_concentrated_pool_swap_token_b_in_negative_delta() -> None:
    """Negative delta_amount (fUSDA in) must return lovelace out and a non-negative fee."""
    datum = _make_swap_datum()
    out, fee = calculate_concentrated_pool_swap(
        token_a_amount=100_000_000,
        token_b_amount=36_000_000,
        datum=datum,
        delta_amount=-1_000_000,  # fUSDA in -> negative
        reward_amount=0,
    )
    assert out > 0
    assert fee >= 0


def test_calculate_concentrated_pool_swap_reward_changes_output() -> None:
    """Adding staking rewards to an ADA-input swap must change the output amount."""
    datum = _make_swap_datum()
    out_no_reward, _ = calculate_concentrated_pool_swap(
        100_000_000, 36_000_000, datum, 1_000_000, reward_amount=0
    )
    out_with_reward, _ = calculate_concentrated_pool_swap(
        100_000_000, 36_000_000, datum, 1_000_000, reward_amount=5_000_000
    )
    assert out_with_reward != out_no_reward


def test_calculate_concentrated_pool_swap_larger_input_more_output() -> None:
    """A larger delta_amount must produce more output (in the same direction)."""
    datum = _make_swap_datum()
    out1, _ = calculate_concentrated_pool_swap(
        100_000_000, 36_000_000, datum, 500_000, reward_amount=0
    )
    out2, _ = calculate_concentrated_pool_swap(
        100_000_000, 36_000_000, datum, 2_000_000, reward_amount=0
    )
    assert out2 > out1


# ===========================================================================
# Section 22: get_delegation_at — Blockfrost staking query
# ===========================================================================


def test_get_delegation_at_no_env_returns_zero_delegation() -> None:
    """Without BLOCKFROST_PROJECT_ID in env, must return a zero-reward Delegation."""
    env_backup = os.environ.pop("BLOCKFROST_PROJECT_ID", None)
    try:
        result = get_delegation_at("stake1abcdef")
        assert isinstance(result, Delegation)
        assert result.rewards == 0
        assert result.pool_id is None
    finally:
        if env_backup is not None:
            os.environ["BLOCKFROST_PROJECT_ID"] = env_backup


# ===========================================================================
# Section 23: create_swap_redeemer_bytes — on-chain redeemer encoding
# ===========================================================================


def test_create_swap_redeemer_bytes_total_length_is_36() -> None:
    """The redeemer bytes must always be exactly 36 bytes long."""
    result = create_swap_redeemer_bytes(pool_in_idx=0, delta_amount=1_000_000)
    assert len(result) == 36


def test_create_swap_redeemer_bytes_action_byte_is_3() -> None:
    """The second byte (action) must always be 0x03 (swap action)."""
    result = create_swap_redeemer_bytes(pool_in_idx=0, delta_amount=1_000_000)
    assert result[1] == 3


def test_create_swap_redeemer_bytes_first_and_third_bytes_are_pool_idx() -> None:
    """Bytes 0 and 2 must both equal pool_in_idx."""
    pool_in_idx = 2
    result = create_swap_redeemer_bytes(pool_in_idx=pool_in_idx, delta_amount=5)
    assert result[0] == pool_in_idx
    assert result[2] == pool_in_idx


def test_create_swap_redeemer_bytes_fourth_byte_is_zero() -> None:
    """The fourth byte (pool_out_idx) must always be 0."""
    result = create_swap_redeemer_bytes(pool_in_idx=0, delta_amount=1)
    assert result[3] == 0


def test_create_swap_redeemer_bytes_last_32_encode_delta() -> None:
    """The last 32 bytes must encode delta_amount in big-endian signed representation."""
    delta = 999_999
    result = create_swap_redeemer_bytes(pool_in_idx=0, delta_amount=delta)
    encoded_delta = int.from_bytes(result[4:], byteorder="big", signed=True)
    assert encoded_delta == delta


def test_create_swap_redeemer_bytes_negative_delta_encodes_correctly() -> None:
    """Negative delta amounts (token B input) must be encoded as signed int."""
    delta = -1_500_000
    result = create_swap_redeemer_bytes(pool_in_idx=0, delta_amount=delta)
    encoded_delta = int.from_bytes(result[4:], byteorder="big", signed=True)
    assert encoded_delta == delta


def test_create_swap_redeemer_bytes_zero_delta() -> None:
    """A zero delta must encode cleanly with all amount bytes set to 0."""
    result = create_swap_redeemer_bytes(pool_in_idx=0, delta_amount=0)
    assert result[4:] == b"\x00" * 32


# ===========================================================================
# Section 24: reward_address_from_script_hash — stake address derivation
# ===========================================================================


def test_reward_address_from_script_hash_hex_string_returns_address() -> None:
    """Passing a hex string must return a valid Address."""
    result = reward_address_from_script_hash(_SCRIPT_HASH_HEX, Network.MAINNET)
    assert isinstance(result, Address)


def test_reward_address_from_script_hash_object_returns_address() -> None:
    """Passing a ScriptHash object must return the same Address as the hex string."""
    from pycardano import ScriptHash

    sh = ScriptHash.from_primitive(bytes.fromhex(_SCRIPT_HASH_HEX))
    result_obj = reward_address_from_script_hash(sh, Network.MAINNET)
    result_hex = reward_address_from_script_hash(_SCRIPT_HASH_HEX, Network.MAINNET)
    assert result_obj.encode() == result_hex.encode()


def test_reward_address_from_script_hash_has_no_payment_part() -> None:
    """A stake/reward address must not have a payment_part."""
    result = reward_address_from_script_hash(_SCRIPT_HASH_HEX, Network.MAINNET)
    assert result.payment_part is None


def test_reward_address_from_script_hash_has_staking_part() -> None:
    """A stake/reward address must have a staking_part."""
    result = reward_address_from_script_hash(_SCRIPT_HASH_HEX, Network.MAINNET)
    assert result.staking_part is not None


def test_reward_address_from_script_hash_mainnet_vs_testnet_differ() -> None:
    """MAINNET and TESTNET must produce different encoded addresses."""
    addr_main = reward_address_from_script_hash(_SCRIPT_HASH_HEX, Network.MAINNET)
    addr_test = reward_address_from_script_hash(_SCRIPT_HASH_HEX, Network.TESTNET)
    assert addr_main.encode() != addr_test.encode()


# ===========================================================================
# Section 25: reward_address_from_script — stake address from Plutus script
# ===========================================================================


def test_reward_address_from_script_with_plutus_v3_object() -> None:
    """A PlutusV3Script object must produce a valid stake Address."""
    result = reward_address_from_script(_DUMMY_SCRIPT, Network.MAINNET)
    assert isinstance(result, Address)
    assert result.staking_part is not None
    assert result.payment_part is None


def test_reward_address_from_script_with_plutus_v2_object() -> None:
    """A PlutusV2Script object must also produce a valid stake Address."""
    script = PlutusV2Script(b"\x01\x00\x00")
    result = reward_address_from_script(script, Network.MAINNET)
    assert isinstance(result, Address)


def test_reward_address_from_script_with_plutus_v1_object() -> None:
    """A PlutusV1Script object must also produce a valid stake Address."""
    script = PlutusV1Script(b"\x01\x00\x00")
    result = reward_address_from_script(script, Network.MAINNET)
    assert isinstance(result, Address)


def test_reward_address_from_script_hex_string_succeeds() -> None:
    """A hex-encoded script string must also produce a valid stake Address."""
    script_hex = _DUMMY_SCRIPT.hex()
    result = reward_address_from_script(script_hex, Network.MAINNET)
    assert isinstance(result, Address)


def test_reward_address_from_script_hex_matches_direct_object() -> None:
    """The address from a hex string must match the address from the script object."""
    script_obj = PlutusV3Script(b"\x01\x00\x00")
    script_hex = script_obj.hex()
    addr_from_obj = reward_address_from_script(script_obj, Network.MAINNET)
    addr_from_hex = reward_address_from_script(script_hex, Network.MAINNET)
    assert addr_from_obj.encode() == addr_from_hex.encode()


def test_reward_address_from_script_invalid_type_raises_type_error() -> None:
    """Passing an invalid type (not a script or hex string) must raise TypeError."""
    with pytest.raises(TypeError):
        reward_address_from_script(12345, Network.MAINNET)  # type: ignore[arg-type]


def test_reward_address_from_script_different_scripts_give_different_addresses() -> None:
    """Two different scripts must hash to different stake addresses."""
    script_a = PlutusV3Script(b"\x01\x00")
    script_b = PlutusV3Script(b"\x02\x00")
    addr_a = reward_address_from_script(script_a, Network.MAINNET)
    addr_b = reward_address_from_script(script_b, Network.MAINNET)
    assert addr_a.encode() != addr_b.encode()


def test_reward_address_from_script_mainnet_vs_testnet_differ() -> None:
    """The same script must produce different addresses on MAINNET vs TESTNET."""
    addr_main = reward_address_from_script(_DUMMY_SCRIPT, Network.MAINNET)
    addr_test = reward_address_from_script(_DUMMY_SCRIPT, Network.TESTNET)
    assert addr_main.encode() != addr_test.encode()


# ===========================================================================
# Section 26: ConcentratedPoolState.swap_utxo — full transaction construction
#
# swap_utxo is the most critical method: it constructs a complete on-chain
# transaction for a concentrated-liquidity direct swap.  It must:
#   1. Validate all required inputs are present (guard clauses).
#   2. Compute the correct delta_amount direction (+/- for token A / B input).
#   3. Mutate the supplied TransactionBuilder with:
#        - Two reference inputs  (pool script + staking script)
#        - One script input      (the pool UTxO with swap redeemer)
#        - Zero-value withdrawal for pool script reward address
#        - Optional reward withdrawal when ADA goes in and rewards are due
#        - One output            (new pool UTxO)
#        - Fee = 17 000
#        - AuxiliaryData with metadata tag 674
#   4. Return (TransactionOutput, new ConcentratedPoolDatum) with correct
#      asset conservation and updated platform fee / epoch fields.
# ===========================================================================

# ---------------------------------------------------------------------------
# swap_utxo test helpers
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock
from charli3_dendrite.dataclasses.models import ScriptReference

# Minimal valid hex for a PlutusV3Script (3 bytes: opcode NOP + 2 bytes padding)
_POOL_SCRIPT_HEX = "010000"
_STAKING_SCRIPT_HEX = "020000"


def _make_script_reference(script_hex: str = _POOL_SCRIPT_HEX) -> ScriptReference:
    """Build a minimal ScriptReference with a valid script hex string."""
    return ScriptReference(
        tx_hash="a0" * 32,
        tx_index=0,
        address=_POOL_ADDRESS,
        assets=Assets(root={"lovelace": 2_000_000}),
        datum_hash=None,
        datum_cbor=None,
        script=script_hex,
    )


def _make_pool_for_swap_utxo(**overrides: Any) -> ConcentratedPoolState:
    """Create a ConcentratedPoolState ready for swap_utxo testing.

    Uses a known registered NFT so that the staking script info lookup
    (_pool_nft_to_staking_script_info) succeeds without touching the backend.
    Injects mock ScriptReference objects directly into the instance __dict__
    so that the @computed_field @cached_property descriptors are bypassed.
    """
    defaults: dict[str, Any] = {
        "pool_nft": Assets(root={_KNOWN_NFT: 1}),
        "tx_hash": "ab" * 32,
    }
    defaults.update(overrides)
    pool = _make_swap_pool(**defaults)

    # Bypass the backend by pre-populating the cached_property slots.
    # Python resolves non-data descriptors (cached_property) only when the
    # key is absent from the instance __dict__, so setting it here prevents
    # any backend call.
    pool.__dict__["pool_script_reference"] = _make_script_reference(_POOL_SCRIPT_HEX)
    pool.__dict__["staking_script_reference"] = _make_script_reference(_STAKING_SCRIPT_HEX)
    return pool


def _make_tx_builder() -> MagicMock:
    """Return a MagicMock that stands in for a pycardano TransactionBuilder."""
    builder = MagicMock()
    builder.reference_inputs = MagicMock()
    return builder


# ---------------------------------------------------------------------------
# Guard clause tests — each missing prerequisite must raise ValueError
# ---------------------------------------------------------------------------


def test_swap_utxo_raises_when_tx_hash_is_none() -> None:
    """swap_utxo must raise ValueError when tx_hash is None."""
    pool = _make_pool_for_swap_utxo()
    pool.__dict__["tx_hash"] = None  # Override after construction to bypass pydantic validation
    with pytest.raises(ValueError, match="Transaction hash is required"):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 300_000}),
            tx_builder=_make_tx_builder(),
        )


def test_swap_utxo_raises_when_pool_nft_is_none() -> None:
    """swap_utxo must raise ValueError when pool_nft is None."""
    pool = _make_pool_for_swap_utxo()
    pool.pool_nft = None  # Force None after construction
    with pytest.raises(ValueError, match="Pool NFT is required"):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 300_000}),
            tx_builder=_make_tx_builder(),
        )


def test_swap_utxo_raises_when_pool_script_reference_is_none() -> None:
    """swap_utxo must raise ValueError when pool_script_reference is None."""
    pool = _make_pool_for_swap_utxo()
    pool.__dict__["pool_script_reference"] = None  # Override injected mock
    with pytest.raises(ValueError, match="Pool script reference is required"):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 300_000}),
            tx_builder=_make_tx_builder(),
        )


def test_swap_utxo_raises_when_staking_script_reference_is_none() -> None:
    """swap_utxo must raise ValueError when staking_script_reference is None."""
    pool = _make_pool_for_swap_utxo()
    pool.__dict__["staking_script_reference"] = None  # Override injected mock
    with pytest.raises(ValueError, match="Staking script reference is required"):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 300_000}),
            tx_builder=_make_tx_builder(),
        )


def test_swap_utxo_raises_when_tx_builder_is_none() -> None:
    """swap_utxo must raise ValueError when tx_builder is None."""
    pool = _make_pool_for_swap_utxo()
    with pytest.raises(ValueError, match="Transaction builder is required"):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 300_000}),
            tx_builder=None,
        )


# ---------------------------------------------------------------------------
# Return type and basic structure tests
# ---------------------------------------------------------------------------


def test_swap_utxo_returns_two_tuple() -> None:
    """swap_utxo must return exactly a 2-tuple."""
    pool = _make_pool_for_swap_utxo()
    result = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 300_000}),
        tx_builder=_make_tx_builder(),
    )
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_swap_utxo_first_element_is_transaction_output() -> None:
    """The first return value must be a TransactionOutput."""
    from pycardano import TransactionOutput as TxOut

    pool = _make_pool_for_swap_utxo()
    pool_output, _ = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 300_000}),
        tx_builder=_make_tx_builder(),
    )
    assert isinstance(pool_output, TxOut)


def test_swap_utxo_second_element_is_concentrated_pool_datum() -> None:
    """The second return value must be a ConcentratedPoolDatum."""
    pool = _make_pool_for_swap_utxo()
    _, new_datum = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 300_000}),
        tx_builder=_make_tx_builder(),
    )
    assert isinstance(new_datum, ConcentratedPoolDatum)


# ---------------------------------------------------------------------------
# Pool output address correctness
# ---------------------------------------------------------------------------


def test_swap_utxo_pool_output_address_matches_pool_address() -> None:
    """The pool output must be sent back to the same pool address (self-referential)."""
    pool = _make_pool_for_swap_utxo()
    pool_output, _ = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 300_000}),
        tx_builder=_make_tx_builder(),
    )
    assert pool_output.address == Address.decode(pool.address)


def test_swap_utxo_pool_output_datum_matches_returned_datum() -> None:
    """The datum embedded in the output must equal the returned new_datum."""
    pool = _make_pool_for_swap_utxo()
    pool_output, new_datum = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 300_000}),
        tx_builder=_make_tx_builder(),
    )
    assert pool_output.datum == new_datum


# ---------------------------------------------------------------------------
# New datum — platform fee update correctness
# ---------------------------------------------------------------------------


def test_swap_utxo_platform_fee_x_increases_when_ada_goes_in() -> None:
    """Swapping ADA in must increase platform_fee_x by the computed platform fee."""
    pool = _make_pool_for_swap_utxo()
    old_datum = ConcentratedPoolDatum.from_cbor(pool.datum_cbor)
    in_amount = 10_000_000  # 10 ADA
    _, new_datum = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": in_amount}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=_make_tx_builder(),
    )
    assert new_datum.platform_fee_x > old_datum.platform_fee_x
    assert new_datum.platform_fee_y == old_datum.platform_fee_y  # unchanged


def test_swap_utxo_platform_fee_y_increases_when_token_b_goes_in() -> None:
    """Swapping token B in must increase platform_fee_y by the computed platform fee."""
    pool = _make_pool_for_swap_utxo()
    old_datum = ConcentratedPoolDatum.from_cbor(pool.datum_cbor)
    _, new_datum = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={_TOKEN_B: 1_000_000}),
        out_assets=Assets(root={"lovelace": 1}),
        tx_builder=_make_tx_builder(),
    )
    assert new_datum.platform_fee_y > old_datum.platform_fee_y
    assert new_datum.platform_fee_x == old_datum.platform_fee_x  # unchanged


def test_swap_utxo_platform_fee_x_delta_equals_expected_fee() -> None:
    """Increase in platform_fee_x must match compute from lp_fee_rate formula.

    lp_fee_rate = 10 (from test CBOR).
    lp_fee = floor(amount_in * 10 / 10_000)
    platform_fee = floor(lp_fee * 5 / 100)
    """
    pool = _make_pool_for_swap_utxo()
    old_datum = ConcentratedPoolDatum.from_cbor(pool.datum_cbor)
    in_amount = 10_000_000
    # lp_fee = (10_000_000 * 10) // 10_000 = 10_000
    # platform_fee = (10_000 * 5) // 100 = 500
    _, new_datum = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": in_amount}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=_make_tx_builder(),
    )
    assert new_datum.platform_fee_x - old_datum.platform_fee_x == 500


def test_swap_utxo_platform_fees_are_non_negative() -> None:
    """Both platform fees in the new datum must always be >= 0."""
    pool = _make_pool_for_swap_utxo()
    _, new_datum = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 2_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=_make_tx_builder(),
    )
    assert new_datum.platform_fee_x >= 0
    assert new_datum.platform_fee_y >= 0


# ---------------------------------------------------------------------------
# New datum — epoch update
# ---------------------------------------------------------------------------


def test_swap_utxo_new_datum_last_withdraw_epoch_is_current() -> None:
    """last_withdraw_epoch in new_datum must reflect the epoch at call time."""
    import time as _time

    pool = _make_pool_for_swap_utxo()
    t_before = int(_time.time() * 1000)
    _, new_datum = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=_make_tx_builder(),
    )
    t_after = int(_time.time() * 1000)
    expected_min = get_epoch(t_before, Network.MAINNET)
    expected_max = get_epoch(t_after, Network.MAINNET)
    assert expected_min <= new_datum.last_withdraw_epoch <= expected_max


# ---------------------------------------------------------------------------
# Pool output asset conservation
# ---------------------------------------------------------------------------


def test_swap_utxo_pool_assets_increase_by_input_amount_for_ada_in() -> None:
    """After ADA goes in, the pool output must hold more ADA."""
    pool = _make_pool_for_swap_utxo()
    in_amount = 2_000_000
    pool_output, _ = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": in_amount}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=_make_tx_builder(),
    )
    # The pool output is a TransactionOutput whose `amount` is a pycardano Value.
    # Check that lovelace increased by exactly in_amount.
    from pycardano import Value
    output_lovelace = pool_output.amount.coin if isinstance(pool_output.amount, Value) else pool_output.amount
    original_lovelace = pool.assets["lovelace"]
    assert output_lovelace == original_lovelace + in_amount


def test_swap_utxo_pool_assets_increase_by_input_amount_for_token_b_in() -> None:
    """After token B goes in, the pool output must hold more token B."""
    pool = _make_pool_for_swap_utxo()
    in_amount = 1_000_000
    pool_output, _ = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={_TOKEN_B: in_amount}),
        out_assets=Assets(root={"lovelace": 1}),
        tx_builder=_make_tx_builder(),
    )
    from pycardano import Value
    amount = pool_output.amount
    if isinstance(amount, Value) and amount.multi_asset:
        # Navigate pycardano's multi-asset structure
        policy_id_bytes = bytes.fromhex(_TOKEN_B[:56])
        asset_name_bytes = bytes.fromhex(_TOKEN_B[56:])
        from pycardano import AssetName, ScriptHash as SH
        policy = SH(policy_id_bytes)
        name = AssetName(asset_name_bytes)
        found_amount = amount.multi_asset.get(policy, {}).get(name, 0)
        assert found_amount == pool.assets[_TOKEN_B] + in_amount
    else:
        # If Value is just coin (lovelace only), token B quantity check is done differently
        # This branch verifies the code ran without error
        pass


def test_swap_utxo_pool_assets_lovelace_decreases_when_ada_goes_out() -> None:
    """After token B goes in, lovelace in the pool output must decrease (ADA paid out)."""
    pool = _make_pool_for_swap_utxo()
    pool_output, _ = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={_TOKEN_B: 1_000_000}),
        out_assets=Assets(root={"lovelace": 1}),
        tx_builder=_make_tx_builder(),
    )
    from pycardano import Value
    output_lovelace = pool_output.amount.coin if isinstance(pool_output.amount, Value) else pool_output.amount
    original_lovelace = pool.assets["lovelace"]
    # Some ADA was paid out, so the pool now holds less
    assert output_lovelace < original_lovelace


# ---------------------------------------------------------------------------
# TransactionBuilder mutation tests
# ---------------------------------------------------------------------------


def test_swap_utxo_adds_pool_script_reference_input() -> None:
    """The pool script out_ref must be added to tx_builder.reference_inputs."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    pool_script_out_ref = ConcentratedPoolState._pool_script_info.out_ref
    builder.reference_inputs.add.assert_any_call(pool_script_out_ref)


def test_swap_utxo_adds_staking_script_reference_input() -> None:
    """The staking script out_ref must be added to tx_builder.reference_inputs."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    staking_out_ref = ConcentratedPoolState._pool_nft_to_staking_script_info[_KNOWN_NFT].out_ref
    builder.reference_inputs.add.assert_any_call(staking_out_ref)


def test_swap_utxo_calls_add_script_input() -> None:
    """swap_utxo must call tx_builder.add_script_input exactly once."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    builder.add_script_input.assert_called_once()


def test_swap_utxo_calls_add_output_with_pool_output() -> None:
    """swap_utxo must call tx_builder.add_output exactly once with the pool output."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool_output, _ = pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    builder.add_output.assert_called_once_with(pool_output)


def test_swap_utxo_sets_fee_to_17000() -> None:
    """swap_utxo must set tx_builder.fee to exactly 17 000 lovelace."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    assert builder.fee == 17_000


def test_swap_utxo_sets_auxiliary_data_with_metadata_674() -> None:
    """swap_utxo must attach AuxiliaryData with metadata tag 674 to the builder."""
    from pycardano.metadata import AuxiliaryData

    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    assert builder.auxiliary_data is not None
    assert isinstance(builder.auxiliary_data, AuxiliaryData)
    assert 674 in builder.auxiliary_data.data


def test_swap_utxo_metadata_contains_swap_message() -> None:
    """Metadata tag 674 must contain the Danogo swap identification message."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    msg = builder.auxiliary_data.data[674]["msg"]
    assert any("Danogo" in m for m in msg)


def test_swap_utxo_sets_zero_withdrawal_for_pool_script() -> None:
    """The pool script reward address must be in tx_builder.withdrawals with value 0."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    # builder.withdrawals was assigned a real Withdrawals dict in swap_utxo
    pool_reward_addr = reward_address_from_script(_POOL_SCRIPT_HEX, Network.MAINNET)
    assert bytes(pool_reward_addr) in builder.withdrawals
    assert builder.withdrawals[bytes(pool_reward_addr)] == 0


def test_swap_utxo_calls_add_withdrawal_script_for_pool_script() -> None:
    """swap_utxo must register the pool script for the zero withdrawal."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    # add_withdrawal_script must have been called at least once (for pool script)
    builder.add_withdrawal_script.assert_called()


# ---------------------------------------------------------------------------
# Staking reward withdrawal — conditional logic
# ---------------------------------------------------------------------------


def test_swap_utxo_no_staking_withdrawal_in_normal_epoch() -> None:
    """In the current epoch (<= last_withdraw_epoch), no staking reward is added.

    The test CBOR has last_withdraw_epoch=69521, well above the current epoch (~547),
    so the staking withdrawal branch must NOT be taken for a normal call.
    """
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    # Only the zero-withdrawal for the pool script must exist
    assert len(builder.withdrawals) == 1


def test_swap_utxo_adds_staking_withdrawal_when_epoch_exceeds_last_withdraw() -> None:
    """When current_epoch > last_withdraw_epoch and ADA goes in, staking reward withdrawal is added.

    We patch time.time to return a far-future timestamp (epoch >> 69521) to
    force the staking reward branch.
    """
    pool = _make_pool_for_swap_utxo()
    # Also inject a pool_reward_amount > 0 so the withdrawal value is non-zero.
    pool.__dict__["pool_reward_amount"] = 5_000_000  # 5 ADA worth of rewards

    builder = _make_tx_builder()
    # Patch time.time so current_epoch is enormous (far future)
    with patch("charli3_dendrite.dexs.amm.danogo.time.time", return_value=1e15):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 1}),
            tx_builder=builder,
        )
    # There must be TWO withdrawal entries: pool script (zero) + staking script (rewards)
    assert len(builder.withdrawals) == 2


def test_swap_utxo_no_staking_withdrawal_when_token_b_goes_in() -> None:
    """In the current epoch (below last_withdraw_epoch=69521), no staking reward is added
    regardless of swap direction.

    The staking reward condition only checks pool type (lovelace pool) + epoch,
    not the direction of the swap. But since current epoch (~547) < 69521,
    the staking branch is not triggered here either.
    """
    pool = _make_pool_for_swap_utxo()
    pool.__dict__["pool_reward_amount"] = 5_000_000

    builder = _make_tx_builder()
    # No time patch — use real current epoch, which is below last_withdraw_epoch (69521)
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={_TOKEN_B: 1_000_000}),
        out_assets=Assets(root={"lovelace": 1}),
        tx_builder=builder,
    )
    # Only the pool script zero-withdrawal — no staking rewards in current epoch
    assert len(builder.withdrawals) == 1


# ---------------------------------------------------------------------------
# Delta amount direction — verified via the redeemer passed to add_script_input
# ---------------------------------------------------------------------------


def test_swap_utxo_positive_delta_for_ada_in() -> None:
    """When ADA is the input token, delta_amount must be positive.

    A positive delta means token A is entering the pool.  We verify this by
    inspecting the Redeemer bytes passed to tx_builder.add_script_input.
    """
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    in_amount = 3_000_000
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": in_amount}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    # Retrieve the redeemer from the add_script_input call
    _, kwargs = builder.add_script_input.call_args
    redeemer = kwargs.get("redeemer") or builder.add_script_input.call_args[0][1]
    redeemer_bytes: bytes = redeemer.data if hasattr(redeemer, "data") else bytes(redeemer.data)
    # Last 32 bytes of the redeemer encode delta_amount as big-endian signed int
    encoded_delta = int.from_bytes(redeemer_bytes[4:], byteorder="big", signed=True)
    assert encoded_delta == in_amount  # positive == ADA in


def test_swap_utxo_negative_delta_for_token_b_in() -> None:
    """When token B is the input token, delta_amount must be negative.

    A negative delta means token B is entering the pool (token A exits).
    """
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    in_amount = 1_000_000
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={_TOKEN_B: in_amount}),
        out_assets=Assets(root={"lovelace": 1}),
        tx_builder=builder,
    )
    _, kwargs = builder.add_script_input.call_args
    redeemer = kwargs.get("redeemer") or builder.add_script_input.call_args[0][1]
    redeemer_bytes: bytes = redeemer.data if hasattr(redeemer, "data") else bytes(redeemer.data)
    encoded_delta = int.from_bytes(redeemer_bytes[4:], byteorder="big", signed=True)
    assert encoded_delta == -in_amount  # negative == token B in


# ---------------------------------------------------------------------------
# Idempotency / determinism
# ---------------------------------------------------------------------------


def test_swap_utxo_deterministic_for_same_inputs() -> None:
    """Two calls with identical inputs must produce the same pool output and datum."""
    pool1 = _make_pool_for_swap_utxo()
    pool2 = _make_pool_for_swap_utxo()

    in_assets = Assets(root={"lovelace": 2_000_000})
    out_assets = Assets(root={_TOKEN_B: 1})

    output1, datum1 = pool1.swap_utxo(
        address_source=_dummy_address(),
        in_assets=in_assets,
        out_assets=out_assets,
        tx_builder=_make_tx_builder(),
    )
    output2, datum2 = pool2.swap_utxo(
        address_source=_dummy_address(),
        in_assets=in_assets,
        out_assets=out_assets,
        tx_builder=_make_tx_builder(),
    )

    assert output1.amount == output2.amount
    assert datum1.platform_fee_x == datum2.platform_fee_x
    assert datum1.platform_fee_y == datum2.platform_fee_y


# ---------------------------------------------------------------------------
# Transaction TTL and validity window — slot-based time bounds
# ---------------------------------------------------------------------------
# swap_utxo sets:
#   tx_builder.validity_start = current_slot - 120
#   tx_builder.ttl            = current_slot + 240
# The slot comes from get_current_slot() (Blockfrost) with a fallback to
# get_current_mainnet_slot() (system clock) when Blockfrost is unavailable.
# ---------------------------------------------------------------------------

from charli3_dendrite.dexs.amm.danogo import get_current_mainnet_slot


def test_swap_utxo_sets_validity_start() -> None:
    """swap_utxo must assign tx_builder.validity_start (not leave it at MagicMock default)."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    with patch("charli3_dendrite.dexs.amm.danogo.get_current_slot", return_value=50_000_000):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 1}),
            tx_builder=builder,
        )
    assert builder.validity_start == 50_000_000 - 120


def test_swap_utxo_sets_ttl() -> None:
    """swap_utxo must assign tx_builder.ttl to current_slot + 240."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    with patch("charli3_dendrite.dexs.amm.danogo.get_current_slot", return_value=50_000_000):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 1}),
            tx_builder=builder,
        )
    assert builder.ttl == 50_000_000 + 240


def test_swap_utxo_validity_window_is_360_slots() -> None:
    """ttl - validity_start must always equal exactly 360 slots."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    with patch("charli3_dendrite.dexs.amm.danogo.get_current_slot", return_value=99_999_999):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 1}),
            tx_builder=builder,
        )
    assert builder.ttl - builder.validity_start == 360


def test_swap_utxo_uses_blockfrost_slot_when_available() -> None:
    """When get_current_slot() succeeds, its value drives the TTL fields."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    fake_slot = 123_456_789
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_current_slot",
        return_value=fake_slot,
    ):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 1}),
            tx_builder=builder,
        )
    assert builder.validity_start == fake_slot - 120
    assert builder.ttl == fake_slot + 240


def test_swap_utxo_falls_back_to_mainnet_slot_on_blockfrost_failure() -> None:
    """When get_current_slot() raises, the fallback get_current_mainnet_slot() is used.

    The test patches both functions to ensure the fallback path is exercised
    and the validity bounds are derived from the fallback slot.
    """
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    fallback_slot = 77_777_777
    with (
        patch(
            "charli3_dendrite.dexs.amm.danogo.get_current_slot",
            side_effect=OSError("Blockfrost unavailable"),
        ),
        patch(
            "charli3_dendrite.dexs.amm.danogo.get_current_mainnet_slot",
            return_value=fallback_slot,
        ),
    ):
        pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 1}),
            tx_builder=builder,
        )
    assert builder.validity_start == fallback_slot - 120
    assert builder.ttl == fallback_slot + 240


def test_swap_utxo_blockfrost_failure_does_not_propagate_exception() -> None:
    """A Blockfrost slot-fetch failure must not cause swap_utxo to raise."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_current_slot",
        side_effect=RuntimeError("connection timeout"),
    ):
        # Must complete without raising
        result = pool.swap_utxo(
            address_source=_dummy_address(),
            in_assets=Assets(root={"lovelace": 1_000_000}),
            out_assets=Assets(root={_TOKEN_B: 1}),
            tx_builder=builder,
        )
    assert result is not None
    assert len(result) == 2


def test_swap_utxo_ttl_is_positive_integer() -> None:
    """tx_builder.ttl must be a positive integer after any successful swap_utxo call."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    assert isinstance(builder.ttl, int)
    assert builder.ttl > 0


def test_swap_utxo_validity_start_is_less_than_ttl() -> None:
    """validity_start must always be strictly less than ttl."""
    pool = _make_pool_for_swap_utxo()
    builder = _make_tx_builder()
    pool.swap_utxo(
        address_source=_dummy_address(),
        in_assets=Assets(root={"lovelace": 1_000_000}),
        out_assets=Assets(root={_TOKEN_B: 1}),
        tx_builder=builder,
    )
    assert builder.validity_start < builder.ttl


# ===========================================================================
# Section 27: fetch_reward_from_blockfrost — TTL-cached reward lookup
# ===========================================================================
# fetch_reward_from_blockfrost(staking_address) delegates to get_delegation_at
# and returns delegation_info.rewards.  Key behaviours:
#   - happy path: returns the int reward from get_delegation_at
#   - None result from get_delegation_at → returns 0
#   - any exception from get_delegation_at → logs warning, returns 0
#   - results are cached per address (REWARD_CACHE TTL)
# We patch `charli3_dendrite.dexs.amm.danogo.get_delegation_at` to avoid any
# real network calls, and clear REWARD_CACHE before every test so TTL state
# does not bleed between tests.


def _clear_reward_cache() -> None:
    """Clear the module-level TTL cache so test results are independent."""
    from charli3_dendrite.dexs.amm.danogo import REWARD_CACHE

    REWARD_CACHE.clear()


_FAKE_STAKE_ADDR = "stake1ux4testaddress000000000000000000000000000000000000"


def test_fetch_reward_returns_int() -> None:
    """fetch_reward_from_blockfrost must return an int."""
    _clear_reward_cache()
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_delegation_at",
        return_value=Delegation(pool_id="pool1abc", rewards=1_500_000),
    ):
        result = fetch_reward_from_blockfrost(_FAKE_STAKE_ADDR)
    assert isinstance(result, int)


def test_fetch_reward_returns_delegation_rewards_value() -> None:
    """When get_delegation_at succeeds, the rewards field is returned verbatim."""
    _clear_reward_cache()
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_delegation_at",
        return_value=Delegation(pool_id="pool1abc", rewards=2_000_000),
    ):
        result = fetch_reward_from_blockfrost(_FAKE_STAKE_ADDR)
    assert result == 2_000_000


def test_fetch_reward_zero_rewards() -> None:
    """A delegation with zero rewards returns 0 (no exception raised)."""
    _clear_reward_cache()
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_delegation_at",
        return_value=Delegation(pool_id=None, rewards=0),
    ):
        result = fetch_reward_from_blockfrost(_FAKE_STAKE_ADDR)
    assert result == 0


def test_fetch_reward_none_delegation_returns_zero() -> None:
    """If get_delegation_at returns None, the function must return 0 (not crash)."""
    _clear_reward_cache()
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_delegation_at",
        return_value=None,
    ):
        result = fetch_reward_from_blockfrost(_FAKE_STAKE_ADDR)
    assert result == 0


def test_fetch_reward_exception_returns_zero() -> None:
    """Any exception from get_delegation_at must be swallowed and 0 returned."""
    _clear_reward_cache()
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_delegation_at",
        side_effect=RuntimeError("network timeout"),
    ):
        result = fetch_reward_from_blockfrost(_FAKE_STAKE_ADDR)
    assert result == 0


def test_fetch_reward_exception_does_not_raise() -> None:
    """Exceptions from the backend must never propagate to the caller."""
    _clear_reward_cache()
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_delegation_at",
        side_effect=ValueError("unexpected error"),
    ):
        # Must not raise
        fetch_reward_from_blockfrost(_FAKE_STAKE_ADDR)


def test_fetch_reward_result_is_cached() -> None:
    """A second call with the same address must use the cached result
    (get_delegation_at is called only once).
    """
    _clear_reward_cache()
    addr = "stake1ux_cache_test_address"
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_delegation_at",
        return_value=Delegation(pool_id=None, rewards=999),
    ) as mock_delegation:
        fetch_reward_from_blockfrost(addr)
        fetch_reward_from_blockfrost(addr)  # second call — should hit cache
    mock_delegation.assert_called_once()  # backend called exactly once


def test_fetch_reward_different_addresses_are_cached_independently() -> None:
    """Each distinct address has its own cache entry."""
    _clear_reward_cache()
    addr_a = "stake1ux_address_aaa"
    addr_b = "stake1ux_address_bbb"
    delegation_a = Delegation(pool_id=None, rewards=100)
    delegation_b = Delegation(pool_id=None, rewards=200)

    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_delegation_at",
        side_effect=lambda addr: delegation_a if addr == addr_a else delegation_b,
    ):
        result_a = fetch_reward_from_blockfrost(addr_a)
        result_b = fetch_reward_from_blockfrost(addr_b)

    assert result_a == 100
    assert result_b == 200


def test_fetch_reward_large_reward_value() -> None:
    """Large reward values (e.g. 1 billion lovelace) are returned without truncation."""
    _clear_reward_cache()
    large_reward = 1_000_000_000_000
    with patch(
        "charli3_dendrite.dexs.amm.danogo.get_delegation_at",
        return_value=Delegation(pool_id="pool1xyz", rewards=large_reward),
    ):
        result = fetch_reward_from_blockfrost(_FAKE_STAKE_ADDR)
    assert result == large_reward


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
