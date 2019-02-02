%% This file produces the results on mixture of 25 Gaussians

clear all;
global covS;
global invS;
V = 1;
% covariance matrix
rho = 0.03;
covS = [ rho, 0; 0, rho];
invS = inv( covS );
% 25 Gaussians 
center = zeros(2,25);
k = 1;
for i = -2:1:2
    for j = -2:1:2
        center(:,k) = [2*i;2*j];
        k = k + 1;
    end
end
p = 0.04;
etaSGLD = 0.043;
etacSGLD = 0.07;
% number of steps 
L = 50000;
% number of cycles
M = 30;

probUMap = @(X,Y) 0*X+0*Y;
func_sum = @(x) [0,0]*x;
func_grad = @(x) 0*x;
func_log = @(x) -log(x);
func_inv = @(x) 1/x;
func_noise = @(x) randn(2,1);
for i = 1:25
    func1 = @(X,Y) p*exp( - 0.5 *( (X-center (1,i)) .* (X-center(1,i)) * invS(1,1) + 2 * (X-center(1,i)).*(Y-center(2,i))*invS(1,2) + (Y-center(2,i)).* (Y-center(2,i)) *invS(2,2) )) / ( 2*pi*sqrt(abs(det (covS))));
    probUMap = @(X,Y) probUMap(X,Y) + func1(X,Y);    
    func2 = @(x) p*exp(-0.5 * (x-center(:,i))'*invS*(x-center(:,i)))/( 2*pi*sqrt(abs(det (covS))));
    func_sum = @(x)func_sum(x) + func2(x);
    sum_inv = @(x)func_inv(func_sum(x));
    func3 = @(x)p*exp(-0.5 * (x-center(:,i))'*invS*(x-center(:,i)))/( 2*pi*sqrt(abs(det (covS))))*invS*(x-center(:,i));
    func_grad = @(x)func_grad(x) + func3(x);
end
grandUTrue = @(x) sum_inv(x) * func_grad(x);
gradUNoise = @(x) grandUTrue(x) + func_noise(x);
[XX,YY] = meshgrid( linspace(-5,5), linspace(-5,5) );
ZZ = probUMap( XX, YY );

randn( 'seed',0 );
% random initialization
x0 = -10 + 20 .* rand(2,1);

dsgld = sgld( gradUNoise, etaSGLD, L, x0, V );
dcsgld = csgld( gradUNoise, etacSGLD, L, M, x0, V );

save('sgld.mat','dsgld');
save('csgld.mat','dcsgld');


