"""
A symmetric arrowhead matrix:

A = [a_1           b_1]
    [    a_2       b_2]
    [       a_3    b_3]
    [           ⋱   ⋮ ]
    [b_1 b_2 b_3 ⋯  c ]
"""
struct SymArrow{T} <: AbstractMatrix{T}
    a::Vector{T}
    b::Vector{T}
    c::T
end

size(A::SymArrow) = length(A.a) + 1, length(A.a) + 1

function getindex(A::SymArrow{T}, i::Integer, j::Integer) where T
    n = size(A, 1)
    if i == j
        if i == n
            return A.c
        else
            return A.a[i]
        end
    elseif i == n
        return A.b[j]
    elseif j == n
        return A.b[i]
    else
        return zero(T)
    end
end

issymmetric(A::SymArrow) = true

+(A::SymArrow, B::SymArrow) = SymArrow(A.a+B.a, A.b+B.b, A.c+B.c)
-(A::SymArrow, B::SymArrow) = SymArrow(A.a-B.a, A.b-B.b, A.c-B.c)

function isincreasing(a::Vector)
    ret = true
    @inbounds for i = 2:length(a)
        ret *= a[i] ≥ a[i-1]
    end
    ret
end

function eig(A::SymArrow{T}) where T
    n = size(A, 1)
    λ = eigvals(A)
    Q = zeros(T, n, n)
    A1 = synthesize_symarrow(A, λ)
    a, b = A1.a, A1.b
    nrmb = norm(b)
    for j = 1:n-1
        λj = λ[j]
        if norm(b[j]) < nrmb*eps(T)
            println("EZ out: ", norm(b[j]) < nrmb*eps(T))
            if norm(a[j] - λj) < norm(a[j] - λ[j+1])
                Q[j,j] = one(T)
            else
                Q[j-1,j] = one(T)
            end
        else
            nrm = zero(T)
            @inbounds @simd for i = 1:n-1
                Q[i,j] = b[i]/(λj-a[i])
                !isfinite(b[i]/(λj-a[i])) && println("j = ",j,", and i = ",i," and ",b[i],"  ",λj,"  ",a[i],"  ",b[i]/(λj-a[i]))
                nrm += abs2(Q[i,j])
            end
            Q[n,j] = one(T)
            nrm += abs2(Q[n,j])
            nrm = sqrt(nrm)
            invnrm = inv(nrm)
            @inbounds @simd for i = 1:n
                Q[i,j] *= invnrm
            end
        end
    end
    λn = λ[n]
    if norm(a[n-1] - λn) < eps(T)
        println("EZ out: ", norm(a[n-1] - λn))
        Q[n-1,n] = one(T)
        #else
        #    Q[n,n] = one(T)
        #end
    else
        nrm = zero(T)
        @inbounds @simd for i = 1:n-1
            Q[i,n] = b[i]/(λn-a[i])
            !isfinite(b[i]/(λn-a[i])) && println("j = ",n,", and i = ",i," and ",b[i],"  ",λn,"  ",a[i],"  ",b[i]/(λn-a[i]))
            nrm += abs2(Q[i,n])
        end
        Q[n,n] = one(T)
        nrm += abs2(Q[n,n])
        nrm = sqrt(nrm)
        invnrm = inv(nrm)
        @inbounds @simd for i = 1:n
            Q[i,n] *= invnrm
        end
    end
    λ, Q
end

"""
Synthesize a symmetric arrow matrix by its eigenvalues and its shaft.
"""
function synthesize_symarrow(A::SymArrow{T}, λ::Vector{T}) where T
    n = size(A, 1)
    a = copy(A.a)
    b = Vector{T}(n-1)
    c = zero(T)
    @inbounds for i = 1:n-1
        c += λ[i] - a[i]
    end
    c += λ[n]

    @inbounds for i = 1:n-1
        ai = a[i]
        t = (ai-λ[1])*(λ[n]-ai)
        for j = 1:i-1
            t *= (λ[j+1]-ai)/(a[j]-ai)
        end
        for j = i+1:n-1
            t *= (λ[j]-ai)/(a[j]-ai)
        end
        b[i] = copysign(sqrt(t), A.b[i])
    end

    SymArrow(a, b, c)
end

function eigvals(A::SymArrow{T}; verbose::Bool = false) where T
    a, b, c = A.a, A.b, A.c
    @assert isincreasing(a)
    n = size(A, 1)
    n2 = n^2
    λ = zeros(T, n)

    nrmb2 = abs(dot(b, b))
    absc = abs(c)

    λk = first_initial_guess(a[1], nrmb2, c)
    δλk = λk
    while abs(δλk) > n2*eps(λk)
        δλk = first_pick_zero_update(A, λk)
        λk += isfinite(δλk) ? δλk : zero(T)
        verbose && println(1," This is the update: ", δλk," and this is the condition: ",n2*eps(λk))
    end
    λ[1] = λk

    for j = 2:n-1
        λk = (a[j]+a[j-1])/2
        δλk = λk
        while abs(δλk) > n2*eps(λk)
            δλk = pick_zero_update(A, j, λk)
            λk += isfinite(δλk) ? δλk : zero(T)
            verbose && println(j," This is the update: ", δλk," and this is the condition: ",n2*eps(λk))
        end
        λ[j] = λk
    end

    λk = last_initial_guess(a[n-1], nrmb2, c)
    δλk = λk
    while abs(δλk) > n2*eps(λk)
        δλk = last_pick_zero_update(A, λk)
        λk += isfinite(δλk) ? δλk : zero(T)
        verbose && println(n," This is the update: ", δλk," and this is the condition: ",n2*eps(λk))
    end
    λ[n] = λk

    return λ
end

function eigvals_FMM(A::SymArrow{T}; verbose::Bool = false) where T
    a, b, c = A.a, A.b, A.c
    @assert isincreasing(a)
    n = size(A, 1)
    n2 = n^2
    λ = zeros(T, n)

    nrmb2 = abs(dot(b, b))
    absc = abs(c)

    λk = first_initial_guess(a[1], nrmb2, c)
    δλk = λk
    while abs(δλk) > n2*eps(λk)
        δλk = first_pick_zero_update(A, λk)
        λk += isfinite(δλk) ? δλk : zero(T)
        verbose && println(1," This is the update: ", δλk," and this is the condition: ",n2*eps(λk))
    end
    λ[1] = λk

    λmidk = (a[2:n-1]+a[1:n-2])/2
    δλk = λmidk
    while norm(δλk) > n2*eps(norm(λmidk))
        δλk = pick_zero_update_FMM(A, λmidk)
        δλk = T[isfinite(δλk) ? δλk : zero(T) for δλk in δλk]
        λmidk .+= δλk
        verbose && println("j This is the update: ", norm(δλk)," and this is the condition: ",n2*eps(norm(λk)))
    end
    λ[2:n-1] .= λmidk

    λk = last_initial_guess(a[n-1], nrmb2, c)
    δλk = λk
    while abs(δλk) > n2*eps(λk)
        δλk = last_pick_zero_update(A, λk)
        λk += isfinite(δλk) ? δλk : zero(T)
        verbose && println(n," This is the update: ", δλk," and this is the condition: ",n2*eps(λk))
    end
    λ[n] = λk

    return λ
end
