function [ data ] = csgld( gradU, eta, L, M, x, V )
%% cSGLD using gradU, for L steps, starting at position x,  
%% return data: array of positions
lr0 = eta;
i=1;
for t = 1 : L
    cos_inner = pi * mod(t-1 , (L / M));
    cos_inner = cos_inner/(L / M);
    cos_out = cos(cos_inner) + 1;
    eta = 0.5*cos_out*lr0;
 
    beta = V * eta * 0.5;
    sigma = sqrt( 2 * eta * (1-beta) );
    dx =  - gradU( x ) * eta + randn(2,1)* sigma;
    x = x + dx;
    if mod(t-1,L/M)+1>50
        data(:,i) = x;
        i = i+1;
    end
end

end
