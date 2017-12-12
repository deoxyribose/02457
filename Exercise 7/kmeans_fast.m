function [resp, C] = kmeans_fast(X, K, par)

[N,d] = size(X);
meanX = mean(X);
try 
    par.covX; 
catch
    par.covX = (X-meanX(ones(N,1),:))'*(X-meanX(ones(N,1),:))/N;
end
try draw = par.draw ; catch draw = 0 ; end


% we need to compute the distance between data and cluster centers
% we divide into three terms, d1 = "X^2", d2 = -2 * C * X, d3 = "C^2" 
d1 = sum( X.^2 , 2 ) ;
oK = ones(K,1) ;
oN = ones(N,1) ;

C=randn(K,d)*chol(par.covX)+ones(K,1)*mean(X) ;

asgn_old = zeros(1,N) ;
asgn = ones(1,N) ;

while sum(abs(asgn_old-asgn))
    
    asgn_old = asgn ;
    
    d3 = sum( C.^2 , 2 ) ;
    
    D = (d1(:,oK))' - 2 * C * X' + d3(:,oN) ;
    
    [val,asgn] = min(D,[],1) ; 

    if draw
        colors = 'r.b.g.m.k.' ;
        colorsx = 'rxbxgxmxkx' ;
        colorso = 'robogomoko' ;
        for k=1:K
            indxk = find(asgn == k) ;
            plot(X(indxk,1),X(indxk,2),colors(1+2*(k-1):2*k)), hold on;
            plot(C(k,1),C(k,2),colorsx(1+2*(k-1):2*k))
            plot(C(k,1),C(k,2),colorso(1+2*(k-1):2*k))
        end
        drawnow
        hold off
        bigfig
        pause
    end
    
    resp = sparse(asgn,1:N,1,K,N) ;
    
    invNK = diag(1 ./ max( 1 , sum( resp, 2 ) ) ) ;
    
    C = invNK * resp * X ;
  
end
