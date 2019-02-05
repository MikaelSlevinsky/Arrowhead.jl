function secular(A::SymArrow{T}, λ::T) where T
    a, b, c = A.a, A.b, A.c
    ret = λ-c
    @inbounds for i = 1:length(a)
        ret += b[i]^2/(a[i]-λ)
    end
    ret
end

function secular_derivative(A::SymArrow{T}, λ::T) where T
    a, b, c = A.a, A.b, A.c
    ret = one(T)
    @inbounds for i = 1:length(a)
        ret += (b[i]/(a[i]-λ))^2
    end
    ret
end

function secular_second_derivative(A::SymArrow{T}, λ::T) where T
    a, b, c = A.a, A.b, A.c
    ret = zero(T)
    @inbounds for i = 1:length(a)
        ret += b[i]^2/(a[i]-λ)^3
    end
    2ret
end

function secular_over_derivative(A::SymArrow{T}, λ::T) where T
    a, b, c = A.a, A.b, A.c
    num = λ-c
    den = one(T)
    @inbounds for i = 1:length(a)
        bi2 = b[i]^2
        aiλ = a[i]-λ
        bi2daiλ = bi2/aiλ
        num += bi2daiλ
        den += bi2daiλ/aiλ
    end
    num/den
end

function pick_parameter(A::SymArrow{T}, j::Integer, λ::T) where T
    f = secular(A, λ)
    fp = secular_derivative(A, λ)
    fpp = secular_second_derivative(A, λ)
    c1 = inv(A.a[j-1]-λ)
    c2 = inv(A.a[j]-λ)
    # α, β, γ
    # γ = (fpp - 2c1*fp)/(2*(c2-c1)*c2^2)
    # β = (fp-c2^2*γ)/c1^2
    # α = f - c1*β - c2*γ
    c2γ = (fpp - 2c1*fp)/(2*(c2-c1)*c2)
    c1β = (fp-c2*c2γ)/c1
    return f - c1β - c2γ
end

function pick_zero_update(A::SymArrow{T}, j::Integer, α::T, λ::T) where T
    f = secular(A, λ)
    fp = secular_derivative(A, λ)
    a = α/((A.a[j-1]-λ)*(λ-A.a[j]))
    b = fp - (inv(A.a[j-1]-λ)+inv(A.a[j]-λ))*f
    c = -f
    return 2c/(b+sqrt(b^2-4a*c))
end

function pick_zero_update(A::SymArrow{T}, j::Integer, λ::T) where T
    f = secular(A, λ)
    fp = secular_derivative(A, λ)
    fpp = secular_second_derivative(A, λ)
    c1 = inv(A.a[j-1]-λ)
    c2 = inv(A.a[j]-λ)
    # α, β, γ
    # γ = (fpp - 2c1*fp)/(2*(c2-c1)*c2^2)
    # β = (fp-c2^2*γ)/c1^2
    # α = f - c1*β - c2*γ
    c2γ = (fpp - 2c1*fp)/(2*(c2-c1)*c2)
    c1β = (fp-c2*c2γ)/c1
    α = f - c1β - c2γ
    a = α/((A.a[j-1]-λ)*(λ-A.a[j]))
    b = fp - (inv(A.a[j-1]-λ)+inv(A.a[j]-λ))*f
    c = -f
    return 2c/(b+sqrt(b^2-4a*c))
end

function first_initial_guess(a1::T, nrmb2::T, c::T) where T
    ret = a1
    if c < a1
        amc2 = (a1-c)/2
        ret -= amc2+sqrt(amc2^2+nrmb2)
    else
        cma2 = (c-a1)/2
        ret -= nrmb2/(cma2+sqrt(cma2^2+nrmb2))
    end
    ret
end

function first_pick_zero_update(A::SymArrow{T}, λ::T) where T
    f = secular(A, λ)
    fp = secular_derivative(A, λ)
    n = length(A.a)
    a1 = A.a[1]
    α = one(T)
    @inbounds for i = 2:n
        α += (A.b[i]/(A.a[i]-λ))^2*(a1-A.a[i])/(λ-A.a[i])
    end
    a = -α/(λ-a1)
    b = fp + f/(λ-a1)
    c = -f
    return 2c/(b+sqrt(b^2-4a*c))
end

function last_initial_guess(an::T, nrmb2::T, c::T) where T
    ret = an
    if c < an
        amc2 = (an-c)/2
        ret += nrmb2/(amc2+sqrt(amc2^2+nrmb2))
    else
        cma2 = (c-an)/2
        ret += cma2+sqrt(cma2^2+nrmb2)
    end
    ret
end

function last_pick_zero_update(A::SymArrow{T}, λ::T) where T
    f = secular(A, λ)
    fp = secular_derivative(A, λ)
    n = length(A.a)
    an = A.a[n]
    α = one(T)
    @inbounds for i = 1:n-1
        α += (A.b[i]/(A.a[i]-λ))^2*(an-A.a[i])/(λ-A.a[i])
    end
    a = -α/(λ-an)
    b = fp + f/(λ-an)
    c = -f
    return 2c/(b+sqrt(b^2-4a*c))
end

# Fast Multipole Method used to accelerate the eigensolve

@inline cauchykernel{T}(x::T, y::T) = inv(x-y)
@inline coulombkernel{T}(x::T, y::T) = inv((x-y)^2)
@inline coulombprimekernel{T}(x::T, y::T) = inv((x-y)^3)
@inline logkernel{T}(x::T, y::T) = log(abs(x-y))

function secular(A::SymArrow{T}, λ::Vector{T}) where T
    a, b, c = A.a, A.b, A.c
    ret = λ-c
    for i = 1:length(λ)
        @inbounds for j = 1:length(a)
            ret[i] += b[j]^2/(a[j]-λ[i])
        end
    end
    ret
end

function secular_FMM(A::SymArrow{T}, λ::Vector{T}) where T
    a, b, c = A.a, A.b, A.c
    amin, amax = extrema(a)
    λmin, λmax = extrema(λ)
    K = KernelMatrix(cauchykernel, reverse(λ), reverse(a), λmax, λmin, amax, amin)
    #ret = λ - c + K*(b.^2)
    ret = λ - c - reverse!(K*(b.^2))
    ret
end

function secular_derivative(A::SymArrow{T}, λ::Vector{T}) where T
    a, b, c = A.a, A.b, A.c
    ret = ones(T, length(λ))
    for i = 1:length(λ)
        @inbounds for j = 1:length(a)
            ret[i] += (b[j]/(a[j]-λ[i]))^2
        end
    end
    ret
end

function secular_derivative_FMM(A::SymArrow{T}, λ::Vector{T}) where T
    a, b, c = A.a, A.b, A.c
    amin, amax = extrema(a)
    λmin, λmax = extrema(λ)
    K = KernelMatrix(coulombkernel, reverse(λ), reverse(a), λmax, λmin, amax, amin)
    ret = ones(T, length(λ)) + K*(b.^2)
    reverse!(ret)
end

function secular_second_derivative(A::SymArrow{T}, λ::Vector{T}) where T
    a, b, c = A.a, A.b, A.c
    ret = zeros(T, length(λ))
    for i = 1:length(λ)
        @inbounds for j = 1:length(a)
            ret[i] += 2b[j]^2/(a[j]-λ[i])^3
        end
    end
    ret
end

function secular_second_derivative_FMM(A::SymArrow{T}, λ::Vector{T}) where T
    a, b, c = A.a, A.b, A.c
    amin, amax = extrema(a)
    λmin, λmax = extrema(λ)
    K = KernelMatrix(coulombprimekernel, reverse(λ), reverse(a), λmax, λmin, amax, amin)
    ret = K*(-2b.^2)
    reverse!(ret)
end

# For all 2 ≤ j ≤ n-1
function pick_zero_update(A::SymArrow{T}, λ::Vector{T}) where T
    n = size(A, 1)
    f = secular(A, λ)
    fp = secular_derivative(A, λ)
    fpp = secular_second_derivative(A, λ)
    c1 = T[inv(A.a[j-1]-λ[j-1]) for j in 2:n-1]
    c2 = T[inv(A.a[j]-λ[j-1]) for j in 2:n-1]
    c2γ = (fpp - 2c1.*fp)./(2*(c2-c1).*c2)
    c1β = (fp-c2.*c2γ)./c1
    α = f - c1β - c2γ
    a = T[α[j-1]/((A.a[j-1]-λ[j-1])*(λ[j-1]-A.a[j])) for j in 2:n-1]
    b = T[fp[j-1] - (inv(A.a[j-1]-λ[j-1])+inv(A.a[j]-λ[j-1]))*f[j-1] for j in 2:n-1]
    c = -f
    return 2c./(b+sqrt.(b.^2-4a.*c))
end

function pick_zero_update_FMM(A::SymArrow{T}, λ::Vector{T}) where T
    n = size(A, 1)
    f = secular_FMM(A, λ)
    fp = secular_derivative_FMM(A, λ)
    fpp = secular_second_derivative_FMM(A, λ)
    c1 = T[inv(A.a[j-1]-λ[j-1]) for j in 2:n-1]
    c2 = T[inv(A.a[j]-λ[j-1]) for j in 2:n-1]
    c2γ = (fpp - 2c1.*fp)./(2*(c2-c1).*c2)
    c1β = (fp-c2.*c2γ)./c1
    α = f - c1β - c2γ
    a = T[α[j-1]/((A.a[j-1]-λ[j-1])*(λ[j-1]-A.a[j])) for j in 2:n-1]
    b = T[fp[j-1] - (inv(A.a[j-1]-λ[j-1])+inv(A.a[j]-λ[j-1]))*f[j-1] for j in 2:n-1]
    c = -f
    return 2c./(b+sqrt.(b.^2-4a.*c))
end
