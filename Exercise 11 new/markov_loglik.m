function loglik = markov_loglik( x,a,K )
%likelihood sequence x with markov transition KxK matrix a 
    nh=x(1:(end-1)) + K*(x(2:end)-1);
    nh=hist(nh,1:(K*K));
    nh=reshape(nh,K,K);
    loglik=sum(sum(nh.*log(a+eps)));
end

