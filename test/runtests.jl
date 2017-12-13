using Arrowhead, Base.Test

srand(0)

import Arrowhead: eigvals_FMM, synthesize_symarrow

@testset "Testing symmetric arrowhead methods" begin
    for n in (100, 300, 1000)

        A = SymArrow((1.0:n-1).*(2.0:n), n*randn(n-1), n*(n+1.0))

        @test issymmetric(A)

        B = SymArrow(ones(n-1), rand(n-1)./n^2, 1.0)

        @test issymmetric(B)

        R = chol(B)

        @test !issymmetric(R)

        @test norm(R'R-B)/norm(B) < n*eps()

        Ri = inv(R)

        @test !issymmetric(Ri)

        @test norm(Ri*R-I) < n*eps()

        SMA = Symmetric(Matrix(A))
        SMB = Symmetric(Matrix(B))

        λ = eigvals(A)
        Λ = eigvals(SMA)
        println("Testing eigenvalue error")
        println(norm(λ-Λ)/norm(Λ),"  ", 20*eps())
        @test norm(λ-Λ)/norm(Λ) < 20*eps()

        A1 = synthesize_symarrow(A, λ)
        Λ, Q = eig(A)

        println("Testing orthogonality")
        println(norm(Q'Q-I),"  ",20*eps())
        #@test norm(Q'Q-I) < 20*eps()

        println("Testing backward error")
        println(norm(A*Q-Q*Diagonal(Λ)),"  ",1.125*(norm(A.b-A1.b)+norm(A.c-A1.c)))
        @test norm(A*Q-Q*Diagonal(Λ)) < 1.125*(norm(A.b-A1.b)+norm(A.c-A1.c))

        λ = eigvals(A, B)
        Λ = eigvals(SMA, SMB)
        println("Testing generalized eigenvalue error")
        println(norm(λ-Λ)/norm(Λ),"  ", 20*eps())
        @test norm(λ-Λ)/norm(Λ) < 20*eps()
    end
end

@testset "Testing FMM acceleration" begin
    for n in (100, 300, 1000, 3000, 10000)
        println("n = ", n)
        A = SymArrow((1.0:n-1).*(2.0:n), n*ones(n-1), n*(n+1.0))
        if n ≤ 10_000
            @time λ = eigvals(A)
            @time λ_FMM = eigvals_FMM(A)

            println("Testing relative error")
            println(norm((λ-λ_FMM)./λ, Inf))
            @test norm((λ-λ_FMM)./λ, Inf) < 10*eps()
        else
            @time eigvals_FMM(A)
        end
    end
end
