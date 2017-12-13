"""
An upper arrowhead matrix:

A = [d_1           e_1]
    [    d_2       e_2]
    [       d_3    e_3]
    [           ⋱   ⋮ ]
    [               f ]
"""
struct UpperArrow{T} <: AbstractMatrix{T}
    d::Vector{T}
    e::Vector{T}
    f::T
end

size(A::UpperArrow) = length(A.d) + 1, length(A.d) + 1

function getindex(A::UpperArrow{T}, i::Integer, j::Integer) where T
    n = size(A, 1)
    if i == j
        if i == n
            return A.f
        else
            return A.d[i]
        end
    elseif j == n
        return A.e[i]
    else
        return zero(T)
    end
end

issymmetric(A::UpperArrow) = false

+(A::UpperArrow, B::UpperArrow) = UpperArrow(A.d+B.d, A.e+B.e, A.f+B.f)
-(A::UpperArrow, B::UpperArrow) = UpperArrow(A.d-B.d, A.e-B.e, A.f-B.f)

function chol(A::SymArrow{T}) where T
    a, b, c = A.a, A.b, A.c
    d = sqrt.(a)
    e = b./d
    f = sqrt(c - norm(e)^2)
    UpperArrow(d, e, f)
end

function inv(A::UpperArrow{T}) where T
    d, e, f = A.d, A.e, A.f
    d1 = inv.(d)
    f1 = inv(f)
    e1 = scale!(-f1, d1.*e)
    UpperArrow(d1, e1, f1)
end

function similarity(A::SymArrow{T}, B::SymArrow{T}) where T
    n = size(A, 1)
    R = chol(B)
    Ri = inv(R)
    a, b, c = A.a, A.b, A.c
    d, e, f = Ri.d, Ri.e, Ri.f
    a1 = Vector{T}(n - 1)
    b1 = Vector{T}(n - 1)
    c1 = c*f^2
    twof = 2f
    @inbounds @simd for i = 1:n-1
        a1[i] = a[i]*d[i]^2
        ae = a[i]*e[i]
        b1[i] = d[i]*(ae + b[i]*f)
        c1 += e[i]*(ae + b[i]*twof)
    end
    SymArrow(a1, b1, c1)
end

eigvals(A::SymArrow{T}, B::SymArrow{T}) where T = eigvals(similarity(A, B))
