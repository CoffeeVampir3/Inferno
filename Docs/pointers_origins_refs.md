# Memory, Ownership, and Pointers in Mojo

## Value semantics

Mojo doesn't enforce value semantics or reference semantics. It supports them both and allows each type to define how it is created, copied, and moved. Mojo is designed with argument behaviors that default to value semantics, and provides tight controls for reference semantics that avoid memory errors. The controls over reference semantics are provided by the value ownership model.

Value semantics generally means that each variable has unique access to a value, and any code outside the scope of that variable cannot modify its value.

In the most basic situation, sharing a value-semantic type means that you create a copy of the value (also known as "pass by value"):

```mojo
def main():
    var x = 1
    var y = x
    y += 1
    print("x:", x)   # x: 1
    print("y:", y)   # y: 2
```

Assigning the value of `x` to `y` creates the value for `y` by making a copy of `x`. Each variable has exclusive ownership of a value. If a type instead used reference semantics, `y` would point to the same value as `x`, and incrementing either one would affect both.

Numeric values in Mojo are value semantic because they're trivial types, which are cheap to copy.

Value semantics also apply to function arguments by default, though the way they apply differs depending on the argument convention. The default behavior for function arguments is fully value semantic: arguments are immutable references, and any living variable from the caller is not affected by the function. Mojo doesn't make any copies unless you explicitly make them yourself.

Reference semantics (mutable references) are also necessary for performant and memory-efficient programs. Rather than enforcing that every variable have "exclusive access" to a value, Mojo ensures that every value has an "exclusive owner," and destroys each value when the lifetime of its owner ends.

## Ownership

The fundamental rules of Mojo's ownership model:

1. Every value has only one owner at a time.
2. When the lifetime of the owner ends, Mojo destroys the value.
3. If there are existing references to a value, Mojo extends the lifetime of the owner.

A variable owns its value. A struct owns its fields. A reference allows you to access a value owned by another variable, with either mutable or immutable access.

Mojo references are created when you call a function: function arguments are passed as mutable or immutable references. A function can return a reference instead of returning a value. To capture a returned reference, use a reference binding:

```mojo
ref value_ref = list[0]
```

### Argument conventions

An argument convention specifies whether an argument is mutable or immutable, and whether the function owns the value. Each convention is defined by a keyword at the beginning of an argument declaration:

- **`read`**: The function receives an immutable reference. It can read the original value (it's not a copy), but can't mutate it.
- **`mut`**: The function receives a mutable reference. It can read and mutate the original value (it's not a copy).
- **`var`**: The function takes ownership of a value. It has exclusive ownership of the argument. The caller might transfer ownership of an existing value, but that's not always what happens. The callee might receive a newly-created value, or a copy of an existing value.
- **`ref`**: The function gets a reference with parametric mutability: the reference is either mutable or immutable. `ref` arguments are a generalization of the `read` and `mut` conventions.
- **`out`**: A special convention used for the `self` argument in constructors and for named results. An `out` argument is uninitialized at the beginning of the function, and must be initialized before the function returns. Although `out` arguments show up in the argument list, they're never passed in by the caller.
- **`deinit`**: A special convention used in the destructor and consuming-move lifecycle methods. A `deinit` argument is initialized at the beginning of the function, and uninitialized when the function returns.

By default, all arguments are `read`.

```mojo
def add(mut x: Int, read y: Int):
    x += y

def main():
    var a = 1
    var b = 2
    add(a, b)
    print(a)   # 3
```

### Immutable arguments (`read`)

The `read` convention is the default. The callee receives an immutable reference to the argument value. Passing an immutable reference is much more efficient when handling large or expensive-to-copy values, because the copy constructor and destructor aren't invoked for a `read` argument.

Mojo's `read` is similar to passing by `const&` in C++, but differs in two important ways:

1. The Mojo compiler implements a lifetime checker that ensures values are not destroyed when there are outstanding references to those values.
2. Small values like `Int`, `Float`, and `SIMD` are always passed in machine registers.

Unlike Rust, Mojo doesn't require a sigil on the caller side to pass by immutable reference, and Rust defaults to moving values instead of borrowing.

### Mutable arguments (`mut`)

To receive a mutable reference, add the `mut` keyword in front of the argument name. Any changes to the value inside the function are visible outside the function:

```mojo
def mutate(mut l: List[Int]):
    l.append(5)
```

Values passed as `mut` must already be mutable. You can't take a `read` value and pass it to another function as `mut`. You can't define default values for `mut` arguments.

### Argument exclusivity

Mojo enforces argument exclusivity for mutable references. If a function receives a mutable reference to a value, it can't receive any other references to the same value—mutable or immutable. A mutable reference can't have any other references that alias it.

```mojo
def append_twice(mut s: String, other: String):
    s += other
    s += other

def invalid_access():
    var my_string = "o"
    # error: passing `my_string` mut is invalid since it's also passed read.
    append_twice(my_string, my_string)
    print(my_string)
```

This code is confusing because the user might expect output `ooo`, but since the first addition mutates both `s` and `other`, the actual output would be `oooo`. To avoid this when you need both, make a copy:

```mojo
var my_string = "o"
var other_string = my_string
append_twice(my_string, other_string)
```

Argument exclusivity isn't enforced for register-passable trivial types (like `Int` and `Bool`) as they're always passed by copy.

### Transfer arguments (`var` and `^`)

To receive value ownership, add the `var` keyword in front of the argument name. Often combined with the postfixed `^` "transfer" sigil on the variable passed in, which ends the lifetime of that variable.

The `var` keyword guarantees only that the function gets unique ownership of a value. This happens in one of three ways:

1. The caller passes the argument with the `^` transfer sigil, which ends the lifetime of that variable (the variable becomes uninitialized) and ownership is transferred into the function.
2. The caller doesn't use the `^` transfer sigil, in which case Mojo copies the value. If the type isn't copyable, this is a compile-time error.
3. The caller passes in a newly-created "owned" value, such as a value returned from a function. No variable owns the value and it's transferred directly to the callee.

```mojo
def take_text(var text: String):
    text += "!"
    print(text)

def main():
    var message = "Hello"
    take_text(message^)
    # print(message)   # error: use of uninitialized value 'message'
```

When the function declares an argument as `var`, it has unique mutable access. Because the value is owned, it is destroyed when the function exits—unless the function transfers it elsewhere.

You shouldn't conflate "ownership transfer" with a "move operation"—these aren't strictly the same. There are multiple ways Mojo transfers ownership:

- If a type implements a take constructor `__init__(out self, *, deinit take: Self)`, Mojo may invoke it when a value is transferred into a function as a `var` argument and the original variable's lifetime ends at the same point.
- In some cases, Mojo optimizes the take entirely away, leaving the value in the same memory location but updating its ownership. In these cases, a value transfers without invoking either the copy or take constructors.

For the `var` convention to work without the transfer sigil, the value type must be copyable (via `__init__(out self, *, copy: Self)`).

## Lifetimes, origins, and references

The Mojo compiler includes a lifetime checker, a compiler pass that analyzes dataflow through your program. It identifies when variables are valid and inserts destructor calls when a variable's lifetime ends.

The compiler uses a special value called an **origin** to track the lifetime of variables and the validity of references. An origin answers two questions:

- What variable "owns" this value?
- Can the value be mutated using this reference?

Origin tracking and lifetime checking is done at compile time. Origins track variables symbolically. Most of the time, origins are handled automatically by the compiler. You'll need to interact with origins directly when working with references (`ref` arguments and `ref` return values), or when working with types like `Pointer` or `Span` which are parameterized on the origin of the data they refer to.

### Origin types

Mojo supplies a struct and a set of type aliases (comptime values) for specifying origin types. `ImmutOrigin` and `MutOrigin` represent immutable and mutable origins:

```mojo
struct ImmutRef[origin: ImmutOrigin]:
    pass
```

The `Origin` struct specifies an origin with parametric mutability:

```mojo
struct ParametricRef[
    is_mutable: Bool,
    //,
    origin: Origin[mut=is_mutable]
]:
    pass
```

Origin types carry the mutability of a reference as a boolean parameter value. The `is_mutable` parameter is an infer-only parameter. The origin value is often inferred as well.

```mojo
from std.memory import Pointer

def use_pointer():
    a = 10
    ptr = Pointer(to=a)
```

### Origin sets

An `OriginSet` is not a type of origin; it represents a group of origins. Origin sets are used for tracking the lifetimes of values captured in closures.

### Origin values

Most origin values are created by the compiler. There are a few ways to specify origin values:

- **Static origin** — `StaticConstantOrigin` represents immutable values that last for the duration of the program. String literal values have a `StaticConstantOrigin`.
- **Derived origin** — `origin_of()` returns the origin associated with the value (or values) passed in.
- **Inferred origin** — inferred parameters can capture the origin of a value passed into a function.
- **External origins** — `MutExternalOrigin` and `ImmutExternalOrigin` represent values that are not tracked by the lifetime checker, such as dynamically-allocated memory.
- **Wildcard origins** — `ImmutAnyOrigin` and `MutAnyOrigin` are special cases indicating a reference that might access any live value.

#### Static origins

Use `StaticConstantOrigin` when you have a value that exists for the entire duration of the program. For example, the `StringLiteral` method `as_string_slice()` returns a `StringSlice` pointing to the original string literal. String literals are static—they're allocated at compile time and never destroyed—so the slice is created with an immutable, static origin.

#### Derived origins

Use the `origin_of(value)` operator to obtain a value's origin. An argument can take an arbitrary expression that yields one of:

- An origin value.
- A value with a memory location.

```mojo
origin_of(self)
origin_of(x.y)
origin_of(foo())
```

`origin_of()` is analyzed statically at compile time; the expressions passed are never evaluated.

```mojo
from std.memory import OwnedPointer, Pointer

struct BoxedString:
    var o_ptr: OwnedPointer[String]

    def __init__(out self, value: String):
        self.o_ptr = OwnedPointer(value)

    def as_ptr(mut self) -> Pointer[String, origin_of(self.o_ptr)]:
        return Pointer(to=self.o_ptr[])
```

The `as_ptr()` method takes its `self` argument as `mut self`. With the default `read` convention, it would be immutable, and the derived origin (`origin_of(self.o_ptr)`) would also be immutable.

You can pass multiple expressions to `origin_of()` to express the union of two or more origins:

```mojo
origin_of(a, b)
```

#### Origin unions

The union of two or more origins creates a new origin that references all of the original origins for the purposes of lifetime extension (so a union of the origins of `a` and `b` extends both lifetimes).

An origin union is mutable if and only if all of its constituent origins are mutable.

A reference whose origin is a subset of a wider union implicitly widens to that union, so branches that produce references with different but compatible origins can be combined into a single return type:

```mojo
def widen_origins(
    a: String, b: String, c: Bool
) -> Pointer[String, origin_of(a, b)]:
    if c:
        return Pointer(to=a)   # Pointer[String, origin_of(a)]
    else:
        return Pointer(to=b)   # Pointer[String, origin_of(b)]
```

Each branch produces a `Pointer` over a single argument's origin; both widen to the declared `origin_of(a, b)` return type without an explicit conversion.

#### Inferred origins

Since origins are parameters, the compiler can infer an origin value from the argument passed to a function or method. This allows a function to return a value that has the same origin as the argument passed to it.

#### External origins

`MutExternalOrigin` and `ImmutExternalOrigin` represent values that do not alias any existing value. They point to memory not owned by any other variable, and are therefore not tracked by the lifetime checker. For example, the `alloc()` function returns an `UnsafePointer` to a new dynamically-allocated block of memory, with the origin `MutExternalOrigin`. When you use an unsafe API like this, you're responsible for managing the lifetime yourself.

#### Wildcard origins

`ImmutAnyOrigin` and `MutAnyOrigin` are special cases indicating a reference that might access any live value. Using a pointer with a wildcard origin into a scope effectively disables Mojo's ASAP destruction for any values in that scope, as long as the pointer is live. Wildcard origins are a last resort — prefer a concrete origin parameter wherever you can express one.

### `ref` arguments

The `ref` convention lets you specify an argument of parametric mutability: you don't need to know in advance whether the passed argument will be mutable or immutable. Reasons to use a `ref` argument:

- You want to accept an argument with parametric mutability.
- You want to tie the lifetime of one argument to the lifetime of another argument.
- You want an argument that is guaranteed to be passed in memory: this can be important and useful for generic arguments that need an identity, irrespective of whether the concrete type is register passable.

The syntax is:

```
ref arg_name: arg_type
```

Or:

```
ref[origin_specifier(s)] arg_name: arg_type
```

In the first form, the origin and mutability of the `ref` argument are inferred from the value passed in. The second form includes an origin clause, consisting of one or more origin specifiers inside square brackets. An origin specifier can be:

- An origin value.
- An arbitrary expression, which is treated as shorthand for `origin_of(expression)`. The following are equivalent:
  ```mojo
  ref[origin_of(self)]
  ref[self]
  ```
- An `AddressSpace` value.
- An underscore character (`_`) to indicate that the origin is unbound. This is equivalent to omitting the origin specifier.

```mojo
def add_ref(ref a: Int, b: Int) -> Int:
    return a + b
```

You can name the origin explicitly. This is useful to restrict the argument to either an `ImmutOrigin` or `MutOrigin`, or to bind a function's return value to the origin of an argument.

The `Span` type is a non-owning view of contiguous data, parameterized on an origin value:

```mojo
from std.collections import List
from std.memory import Span

def to_byte_span[
    is_mutable: Bool,
    //,
    origin: Origin[mut=is_mutable],
](ref[origin] list: List[Byte]) -> Span[Byte, origin]:
    return Span(list)

def main():
    list: List[Byte] = [77, 111, 106, 111]
    _ = to_byte_span(list)
```

The `origin` parameter is inferred from the `list` argument, then used as the origin for the returned `Span`. The span will have the same lifetime as the list, and will be mutable if the list is mutable.

### `ref` return values

The syntax for a `ref` return value:

```
-> ref[origin_specifier(s)] arg_type
```

You **must** provide an origin specifier for a `ref` return value. The values allowed are the same as for `ref` arguments.

`ref` return values can be an efficient way to handle updating items in a collection. With a `ref` argument, `__getitem__()` can return a mutable reference that can be modified directly. Pros and cons compared to using `__setitem__()`:

- The mutable reference is more efficient—a single update isn't broken up across two methods. However, the referenced value must be in memory.
- A `__getitem__()`/`__setitem__()` pair allows for arbitrary code to be run when values are retrieved and set. For example, `__setitem__()` can validate or constrain input values.

```mojo
struct NameList:
    var names: List[String]

    def __init__(out self, *names: String):
        self.names = []
        for name in names:
            self.names.append(name)

    def __getitem__(ref self, index: Int) raises -> ref[self.names] String:
        if (index >= 0 and index < len(self.names)):
            return self.names[index]
        else:
            raise Error("index out of bounds")

def main() raises:
    list = NameList("Thor", "Athena", "Dana", "Vrinda")
    ref name = list[2]
    print(name)        # Dana
    name += "?"
    print(list[2])     # Dana?
```

If you assign a `ref` return value to a variable, the variable receives a copy of the referenced item. Use a reference binding to capture the reference:

```mojo
var name_copy = list[2]   # owned copy of list[2]
ref name_ref = list[2]    # reference to list[2]
```

### Parametric mutability of return values

Since the origin of the return value can be tied to the origin of `self`, the returned reference will be mutable if the method was called using a mutable reference. The method still works with an immutable reference to the receiver, but returns an immutable reference. Without parametric mutability, you'd need to write two versions of `__getitem__()`, one for immutable `self` and another for mutable `self`.

### Return values with union origins

A `ref` return value can include multiple values in its origin specifier, yielding the union of the origins:

```mojo
def pick_one(cond: Bool, ref a: String, ref b: String) -> ref[a, b] String:
    return a if cond else b
```

Because the compiler can't statically determine which branch will be picked, this function must use the union origin `[a, b]`. This ensures the compiler extends the lifetime of both values as long as the returned reference is live. The returned reference is mutable if both `a` and `b` are mutable.

### Parameterized `out`

A type can parameterize the address space of its `out` self, letting a constructor be bound at the call site for any (or a specific) address space:

```mojo
struct MemType(Movable):
    # Constructable into any address space.
    def __init__[addr_space: AddressSpace](out[addr_space] self):
        ...

    # Only constructable into AddressSpace.GLOBAL.
    def __init__(arg: Int, out[AddressSpace.GLOBAL] self):
        ...
```

This is how a type expresses that some constructors are address-space-polymorphic while others are pinned to a single space (for example, global memory only).

## Pointers

A pointer is an indirect reference to one or more values stored in memory. The pointer is a value that holds an address to memory, and provides APIs to store and retrieve values to that memory. The value pointed to by a pointer is also known as a pointee.

The Mojo standard library includes several types of pointers, which provide different sets of features. All pointer types are generic—they can point to any type of value, with the value type specified as a parameter:

```mojo
from std.memory import OwnedPointer

var ptr: OwnedPointer[Int]
ptr = OwnedPointer(100)
```

Accessing the memory—to retrieve or update a value—is called dereferencing the pointer. Dereference by following the variable name with empty square brackets:

```mojo
ptr[] += 10
print(ptr[])
```

### Pointer terminology

- **Safe pointers**: designed to prevent memory errors. Unless you use APIs specially designated as unsafe, you can use these pointers without worrying about issues like double-free or use-after-free.
- **Nullable pointers**: can point to an invalid memory location (typically 0, or a "null pointer"). None of the standard library pointer types are nullable. To model a nullable pointer, use `Optional[UnsafePointer]`. See [Modeling absence vs split initialization](#modeling-absence-vs-split-initialization) for guidance on the two distinct intents this can express.
- **Smart pointers**: own their pointees, which means the value they point to may be deallocated when the pointer itself is destroyed. Non-owning pointers may point to values owned elsewhere, or may require some manual management of the value lifecycle.
- **Memory allocation**: some pointer types can allocate memory to store their pointees, while others can only point to pre-existing values. Allocation can be implicit (performed automatically when initializing a pointer with a value) or explicit.
- **Uninitialized memory**: memory locations that haven't been initialized with a value, which may contain random data. Newly-allocated memory is uninitialized. Safe pointer types don't allow access to uninitialized memory. Unsafe pointers can allocate a block of uninitialized memory and initialize them one at a time. Accessing uninitialized memory is unsafe by definition.
- **Copyable types**: can be copied implicitly (e.g., by assigning a value to a variable). Also called implicitly copyable types.
- **Explicitly copyable types**: require the user to request a copy, using a constructor with a keyword argument:
  ```mojo
  copied_owned_ptr = OwnedPointer(other=owned_ptr)
  ```

### Pointer types

| | `Pointer` | `OwnedPointer` | `ArcPointer` | `UnsafePointer` |
|---|---|---|---|---|
| Safe | Yes | Yes | Yes | No |
| Allocates memory | No | Implicitly¹ | Implicitly¹ | Explicitly |
| Owns pointee(s) | No | Yes | Yes | No² |
| Copyable | Yes | No³ | Yes | Yes |
| Nullable | No | No | No | No |
| Can point to uninitialized memory | No | No | No | Yes |
| Can point to multiple values (array-like access) | No | No | No | Yes |

¹ `OwnedPointer` and `ArcPointer` implicitly allocate memory when you initialize the pointer with a value.

² `UnsafePointer` provides unsafe methods for initializing and destroying instances of the stored type. The user is responsible for managing the lifecycle of stored values.

³ `OwnedPointer` is explicitly copyable, but explicitly copying an `OwnedPointer` copies the stored value into a new `OwnedPointer`.

### `Pointer`

The `Pointer` type is a safe pointer that points to an initialized value that it doesn't own. Example use cases:

- Storing a reference to a related type. For example, a list's iterator object might hold a `Pointer` back to the original list.
- Passing the memory location for a single value to external code via `external_call()`.
- Where you need an API to return a long-lived reference to a value. (Currently the iterators for standard library collection types like `List` return pointers.)

Construct a `Pointer` to an existing value with the `to` keyword argument:

```mojo
from std.memory import Pointer

ptr = Pointer(to=some_value)
```

You can also create a `Pointer` by copying an existing `Pointer`. A `Pointer` carries an origin for the stored value, so Mojo can track the lifetime of the referenced value.

### `OwnedPointer`

The `OwnedPointer` type is a smart pointer designed for cases where there is single ownership of the underlying data. An `OwnedPointer` points to a single item, passed in when you initialize the `OwnedPointer`. The `OwnedPointer` allocates memory and moves or copies the value into the reserved memory:

```mojo
from std.memory import OwnedPointer

o_ptr = OwnedPointer(some_big_struct)
```

An owned pointer can hold almost any type of item, but when constructing an `OwnedPointer`, the stored item must be either `Movable` or `Copyable`. Since `OwnedPointer` is designed to enforce single ownership, the pointer itself can be moved, but not copied.

Self-referential structures like linked lists and trees that need an "absent" state can spell it as `Optional[OwnedPointer[T]]`. The `Optional` is `None` when the slot is empty and holds a moved-in `OwnedPointer` when populated.

### `ArcPointer`

An `ArcPointer` is a reference-counted smart pointer, ideal for shared resources where the last owner for a given value may not be clear. Like an `OwnedPointer`, it points to a single value, and it allocates memory when you initialize the `ArcPointer` with a value:

```mojo
from std.memory import ArcPointer

attributesDict: Dict[String, String] = {}
attributes = ArcPointer(attributesDict)
```

Unlike an `OwnedPointer`, an `ArcPointer` can be freely copied. All instances of a given `ArcPointer` share a reference count, which is incremented whenever the `ArcPointer` is copied and decremented whenever an instance is destroyed. When the reference count reaches zero, the stored value is destroyed and the allocated memory is freed.

You can use `ArcPointer` to implement safe reference-semantic types:

```mojo
from std.memory import ArcPointer

struct SharedDict(Copyable):
    var attributes: ArcPointer[Dict[String, String]]

    def __init__(out self):
        attributesDict: Dict[String, String] = {}
        self.attributes = ArcPointer(attributesDict)

    def __init__(out self, *, copy: Self):
        self.attributes = copy.attributes

    def __setitem__(mut self, key: String, value: String):
        self.attributes[][key] = value

    def __getitem__(self, key: String) -> String:
        return self.attributes[].get(key, default="")

def main():
    thing1 = SharedDict()
    thing2 = thing1
    thing1["Flip"] = "Flop"
    print(thing2["Flip"])
```

The reference count is stored using an `Atomic` value, so updates to the count itself are thread-safe. Mojo does not enforce argument exclusivity across thread boundaries, so user code is responsible for synchronizing access to the *referenced value*.

### `UnsafePointer`

`UnsafePointer` is a low-level pointer that can access a block of contiguous memory locations, which might be uninitialized. It's analogous to a raw pointer in C and C++. `UnsafePointer` provides unsafe methods for initializing and destroying stored values, as well as for accessing the values once they're initialized.

`UnsafePointer` doesn't provide any memory safety guarantees, so you should reserve it for cases when none of the other pointer types will do the job. Use cases:

- Building a high-performance array-like structure, such as `List` or `Tensor`. A single `UnsafePointer` can access many values, and gives you a lot of control over how you allocate, use, and deallocate memory. Being able to access uninitialized memory means you can preallocate a block of memory, and initialize values incrementally as they are added to the collection.
- Interacting with external libraries including C++ and Python. You can use `UnsafePointer` to pass a buffer full of data to or from an external library.

#### Modeling absence vs split initialization

A field or parameter that "doesn't have a real pointer yet" can mean two different things. Pick the form that matches what you mean:

- **Split / delayed initialization** — the field will be assigned a real pointer before any dereference. Use `UnsafePointer[T, origin].unsafe_dangling()`. The value is a placeholder; never deref it before assigning. Common in `__init__` bodies that allocate after some preliminary setup, or that may take an early-return path before allocation succeeds (in which case access must be gated by a separate validity flag).

- **Genuinely nullable** — absence is a meaningful state the caller or callee can test for. Use `Optional[UnsafePointer[T, origin]]`, defaulting to `None` where appropriate. Check with `if opt:` or `== None`; unwrap with `.value()` when present. This is the right choice for default parameters where "I don't want this" is a real option (e.g. syscall `old=` outparams), or for fields that may permanently lack a pointer.

```mojo
from std.memory import UnsafePointer, alloc

# Split init: field allocated later in __init__
struct Pool:
    var buf: UnsafePointer[Byte, MutAnyOrigin]

    def __init__(out self, n: Int):
        self.buf = UnsafePointer[Byte, MutAnyOrigin].unsafe_dangling()
        # ... preliminary work ...
        self.buf = alloc[Byte](n)

# Nullable: caller may pass nothing
def sigaltstack(
    ss: UnsafePointer[StackT, MutAnyOrigin],
    old: Optional[UnsafePointer[StackT, MutAnyOrigin]] = None,
) -> Int:
    return syscall(NR_sigaltstack, Int(ss),
                   Int(old.value()) if old else 0)
```

Two design smells indicate you've picked the wrong one:

- Reaching for `unsafe_dangling()` *and* writing a separate "is this valid yet?" flag in the same struct → the field is actually `Optional`.
- Writing `Optional[UnsafePointer[...]]` and unwrapping it with `.value()` on every hot-path access → it's actually split init, and the validity should be established once by a separate gate so the hot path can use the raw pointer.
