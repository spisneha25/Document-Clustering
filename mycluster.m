function [ class ] = mycluster( D, k )
    %
    % Your goal of this assignment is implementing your own text clustering algo.
    %
    % Input:
    %     bow: data set. Bag of words representation of text document as
    %     described in the assignment.
    %
    %     K: the number of desired topics/clusters. 
    %
    % Output:
    %     class: the assignment of each topic. The
    %     assignment should be 1, 2, 3, etc. 
    %
    % For submission, you need to code your own implementation without using
    % any existing libraries

    % YOUR IMPLEMENTATION SHOULD START HERE!

    %%%Initialize the EM Algorithm parameters%%%
    threshold = 1/100000;
    maxiter = 50;
    [W, M] = Initialize(D, k);
    Ln = Likelihood(D, k, W, M);
    Lo = 2*Ln;

    %%%EM Algorithm%%%
    iter = 0;
    improvement = abs(100 * (Ln - Lo)/Lo);
    while (improvement > threshold) && (iter <= maxiter),
        E = Expectation(D, k, W, M);
        [W, M] = Maximization(D, k, E);
        Lo = Ln;
        Ln = Likelihood(D, k, W, M);
        iter = iter + 1;
        improvement = abs(100 * (Ln - Lo)/Lo);
    end

    %%%Expectation Function%%%
    function E = Expectation(X, k, W, M)
        [n, d] = size(X);
        den = zeros(n);
        E = zeros(n, k);
        for i = 1:n
            for c = 1:k
                num(i, c) = W(c)*prod(M(:, c).^(X(i, :)'));
                den = sum(num, 2);
            end
            E(i, :) = rdivide(num(i,:), den(i)); 
            [r, class] = max(E, [], 2);
        end
    end

    %%%Maximization Function%%%
    function [W, M] = Maximization(X, k, E)
        [n, d] = size(X);
        W = zeros(1, k);
        M = zeros(d, k);
        num = E'*X;
        den = sum(num, 2); 
        for c = 1:k
            W(c) = sum(E(:, c))/n;
            M(:, c) = rdivide(num(c, :),den(c));
        end
    end

    function L = Likelihood(X, k, W, M)
        L = 0;
        [n, d] = size(X);
        v1 = zeros(k);
        v2 = zeros(n);
        for i = 1 : n
            for c = 1 : k
                v1(c) = 1;
                for j = 1 : d
                    v1(c) = v1(c) * M(j, c)^ X(i, j);
                end
                v2(i) = v2(i) + W(c)*v1(c);
            end
            L = L + log(v2(i));
        end  
    end

    function [W, M] = Initialize(X, k)
        [n, d] = size(X);
        Nz = rand(k,d); 
        S = sum(Nz, 2);
        for i = 1:k
            Nz(i, :) = Nz(i, :)/S(i);
        end
        M = Nz'; 
        W = rand(1, k);
        W = W/sum(W);
    end
end

