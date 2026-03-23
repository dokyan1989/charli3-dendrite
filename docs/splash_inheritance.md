# Splash DEX — Inheritance Hierarchy

## Abstract Base Classes

```plain
DendriteBaseModel (pydantic) + ABC
└── AbstractPairState                              core/base.py
    └── AbstractPoolState                          amm/amm_base.py
        ├── AbstractConstantProductPoolState       amm/amm_types.py
        └── AbstractStableSwapPoolState            amm/amm_types.py
            └── AbstractCommonStableSwapPoolState  amm/amm_types.py
```

---

## Pool State Classes

```plain
AbstractPairState
└── SplashBaseState
    │     (dex name, order selector, NFT/LP extraction)
    │
    ├── SplashSSPState
    │     (SplashBaseState + AbstractCommonStableSwapPoolState)
    │     Stable swap pool — uses SSPoolRedeemer
    │
    └── SplashCPPState
          (SplashBaseState + AbstractConstantProductPoolState)
          Constant product pool — uses CPPoolRedeemer
          │
          ├── SplashCPPBidirState
          │     Bidirectional fees (pool_fee_x / pool_fee_y)
          │
          └── SplashCPPRoyaltyState
                Adds royalty fee accounting (royalty_x / royalty_y)
```

---

## Datum & Redeemer Classes

### Pool Datums

```plain
PoolDatum
├── SplashSSPPoolDatum       Stable swap pool datum (an2n, multipliers, protocol fees)
├── SplashCPPPoolDatum       Constant product pool datum (pool_fee, treasury)
├── SplashCPPBidirPoolDatum  Bidirectional CPP datum (pool_fee_x, pool_fee_y)
└── SplashCPPRoyaltyPoolDatum  Royalty CPP datum (royalty_fee, royalty_x/y)
```

### Order Datum

```plain
OrderDatum
└── SplashOrderDatum         Swap order (tag, beacon, price, fees, redeemer address)
```

### Redeemers & Actions

```plain
PlutusData
├── BoolFalse / BoolTrue     Boolean wrappers
├── Rationale                Numerator / denominator fraction
├── SwapAction               Carries context_values_list (D invariant)
├── PDAOAction               DAO governance action
├── SSPoolRedeemer           Redeemer for stable swap pool (pool_in_idx, pool_out_idx, action)
└── CPPoolRedeemer           Redeemer for constant product pool (action, self_index)
```

---

## MRO Summary

| Concrete Class | Resolution Order |
| --- | --- |
| `SplashSSPState` | `SplashBaseState` → `AbstractCommonStableSwapPoolState` → `AbstractStableSwapPoolState` → `AbstractPoolState` → `AbstractPairState` |
| `SplashCPPState` | `SplashBaseState` → `AbstractConstantProductPoolState` → `AbstractPoolState` → `AbstractPairState` |
| `SplashCPPBidirState` | `SplashCPPState` → *(same as above)* |
| `SplashCPPRoyaltyState` | `SplashCPPState` → *(same as above)* |
