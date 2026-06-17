from kernels.helpers import Binding
from quant.recipe import (
    QuantRecipe, PerRowQuant, PerBlockQuant,
    PerRowCs, PerBlockCs, VnniPacked,
)


@always_inline
def quant_k_block[q: QuantRecipe]() -> Int:
    comptime if q.isa[PerRowQuant]():
        return q[PerRowQuant].fwht_block
    comptime if q.isa[PerBlockQuant]():
        return q[PerBlockQuant].fwht_block
    return 0


@always_inline
def quant_per_block[q: QuantRecipe]() -> Bool:
    return q.isa[PerBlockQuant]()


@always_inline
def quant_vnni_packed[q: QuantRecipe]() -> Bool:
    comptime if q.isa[PerRowQuant]():
        return q[PerRowQuant].pack.isa[VnniPacked]()
    comptime if q.isa[PerBlockQuant]():
        return q[PerBlockQuant].pack.isa[VnniPacked]()
    return False


@always_inline
def quant_colsum_per_block[q: QuantRecipe]() -> Bool:
    comptime if q.isa[PerRowQuant]():
        return q[PerRowQuant].colsum.isa[PerBlockCs]()
    comptime if q.isa[PerBlockQuant]():
        return q[PerBlockQuant].colsum.isa[PerBlockCs]()
    return False


@always_inline
def quant_has_colsum[q: QuantRecipe]() -> Bool:
    comptime if q.isa[PerRowQuant]():
        comptime QT = q[PerRowQuant]
        return QT.colsum.isa[PerRowCs]() or QT.colsum.isa[PerBlockCs]()
    comptime if q.isa[PerBlockQuant]():
        return q[PerBlockQuant].colsum.isa[PerBlockCs]()
    return False


@fieldwise_init
struct ButterquantWeight[quant: QuantRecipe, o: ImmutOrigin](
    Copyable, ImplicitlyCopyable,
):
    comptime HAS_COLSUM = quant_has_colsum[Self.quant]()

    var data: Binding[Int8, Self.o]
    var scale: Binding[Float32, Self.o]
    var colsum: Binding[Float32, Self.o]

    @always_inline
    def colsum_checked(self) -> Binding[Float32, Self.o]:
        comptime assert Self.HAS_COLSUM, (
            "ButterquantWeight recipe declared no colsum but tried to access one.")
        return self.colsum


@fieldwise_init
struct ButterquantRouter[quant: QuantRecipe, o: ImmutOrigin](
    Copyable, ImplicitlyCopyable,
):
    var centered: Binding[BFloat16, Self.o]
    var gauge: Optional[Binding[BFloat16, Self.o]]
    var bias: Optional[Binding[Float32, Self.o]]


@fieldwise_init
struct ButterquantActivation[o: ImmutOrigin](Copyable, ImplicitlyCopyable):
    var data: Binding[Int8, Self.o]
    var scale: Binding[Float32, Self.o]


@fieldwise_init
struct ButterquantBlockActivation[o: ImmutOrigin](Copyable, ImplicitlyCopyable):
    var data: Binding[Int8, Self.o]
    var scale: Binding[Float32, Self.o]
