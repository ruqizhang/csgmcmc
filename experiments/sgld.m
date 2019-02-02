function [ data ] = sgld( gradU, eta, L, x, V )
%% SGLD using gradU, for L steps, starting at position x, use SGFS way to take noise gradient level into account, 
%% return data: array of positions
i = 1;
a = eta;
b = 0;
gamma = 0.55;
iter = 500;
for t = 1 : L
    if t<iter+1
        eta = a;
    else
        eta = a*power(b + t-iter,-gamma);
    end
    beta = V * eta * 0.5;
    sigma = sqrt( 2 * eta * (1-beta) );
    dx = - gradU( x ) * eta + randn(2,1) * sigma;
    x = x + dx;
    data(:,i) = x;
    i = i+1;
end

end
