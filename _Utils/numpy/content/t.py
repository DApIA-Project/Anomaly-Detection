
class _ArrayOrScalarCommon:
    @property
    def T(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def data(self) -> memoryview: ...
    @property
    def flags(self) -> flagsobj: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __copy__(self: _ArraySelf) -> _ArraySelf: ...
    def __deepcopy__(self: _ArraySelf, memo: None | dict[int, Any], /) -> _ArraySelf: ...

    # TODO: How to deal with the non-commutative nature of `==` and `!=`?
    # xref numpy/numpy#17368
    def __eq__(self, other: Any) -> Any: ...
    def __ne__(self, other: Any) -> Any: ...
    def copy(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    def dump(self, file: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _SupportsWrite[bytes]) -> None: ...
    def dumps(self) -> bytes: ...
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...
    # NOTE: `tostring()` is deprecated and therefore excluded
    # def tostring(self, order=...): ...
    def tofile(
        self,
        fid: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _IOProtocol,
        sep: str = ...,
        format: str = ...,
    ) -> None: ...
    # generics and 0d arrays return builtin scalars
    def tolist(self) -> Any: ...

    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    @property
    def __array_priority__(self) -> float: ...
    @property
    def __array_struct__(self) -> Any: ...  # builtins.PyCapsule
    def __setstate__(self, state: tuple[
        SupportsIndex,  # version
        _ShapeLike,  # Shape
        _DType_co,  # DType
        bool,  # F-continuous
        bytes | list[Any],  # Data
    ], /) -> None: ...
    # a `bool_` is returned when `keepdims=True` and `self` is a 0d array

    @overload
    def all(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: L[False] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> bool_: ...
    @overload
    def all(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def all(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def any(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: L[False] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> bool_: ...
    @overload
    def any(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def any(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def argmax(
        self,
        axis: None = ...,
        out: None = ...,
        *,
        keepdims: L[False] = ...,
    ) -> intp: ...
    @overload
    def argmax(
        self,
        axis: SupportsIndex = ...,
        out: None = ...,
        *,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmax(
        self,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def argmin(
        self,
        axis: None = ...,
        out: None = ...,
        *,
        keepdims: L[False] = ...,
    ) -> intp: ...
    @overload
    def argmin(
        self,
        axis: SupportsIndex = ...,
        out: None = ...,
        *,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmin(
        self,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    def argsort(
        self,
        axis: None | SupportsIndex = ...,
        kind: None | _SortKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> ndarray: ...

    @overload
    def choose(
        self,
        choices: ArrayLike,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray: ...
    @overload
    def choose(
        self,
        choices: ArrayLike,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: None | ArrayLike = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> ndarray: ...
    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> ndarray: ...
    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: None | ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass: ...
    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass: ...

    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: None | SupportsIndex = ...,
        out: None = ...,
    ) -> ndarray: ...
    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    def conj(self: _ArraySelf) -> _ArraySelf: ...

    def conjugate(self: _ArraySelf) -> _ArraySelf: ...

    @overload
    def cumprod(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumprod(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def cumsum(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumsum(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def max(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def max(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def mean(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def mean(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def min(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def min(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    def newbyteorder(
        self: _ArraySelf,
        __new_order: _ByteOrder = ...,
    ) -> _ArraySelf: ...

    @overload
    def prod(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def prod(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def ptp(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def ptp(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def round(
        self: _ArraySelf,
        decimals: SupportsIndex = ...,
        out: None = ...,
    ) -> _ArraySelf: ...
    @overload
    def round(
        self,
        decimals: SupportsIndex = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def std(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def std(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def sum(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def sum(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def var(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def var(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...