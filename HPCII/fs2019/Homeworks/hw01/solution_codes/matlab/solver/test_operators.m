clc; clear

addpath('./solver')

c1 = 1;c2 = 0.01;c3 = 0.5;c4 = 0.5;

% f = @(X,Y) (4*c1*exp(-((c3 - X).^2 + (c4 - Y).^2)/c2).*(c3.^2 - 2*c3*X + c4^2 - 2*c4*Y + X.^2 + Y.^2 - c2))/c2^2;
% f = @(X,Y) X.^2+Y.^2;
f = @(X,Y) (X.^2-X.^4).*(Y.^4-Y.^2);

k1 = 5;
k2 = 4;

N1 = 2^k1+1;
x1 = linspace(-1,1,N1);
[X1,Y1] = meshgrid(x1,x1);
Z1e = f(X1,Y1);



N2 = 2^k2+1;
x2 = linspace(-1,1,N2);
[X2,Y2] = meshgrid(x2,x2);
Z2e = f(X2,Y2);

Z1 = prolongate( Z2e, N2, N1);
Z2 = restrict( Z1e, N2 );



subplot(3,2,1)
surf(X1(2:end-1,2:end-2),Y1(2:end-1,2:end-2),Z1e(2:end-1,2:end-2))
colorbar

subplot(3,2,2)
surf(X2(2:end-1,2:end-2),Y2(2:end-1,2:end-2),Z2e(2:end-1,2:end-2))
colorbar

subplot(3,2,3)
surf(X1(2:end-1,2:end-2),Y1(2:end-1,2:end-2),Z1(2:end-1,2:end-2))
colorbar

subplot(3,2,4)
surf(X2(2:end-1,2:end-2),Y2(2:end-1,2:end-2),Z2(2:end-1,2:end-2))
colorbar

subplot(3,2,5)
e1 = (Z1-Z1e);
surf(X1(2:end-1,2:end-2),Y1(2:end-1,2:end-2),e1(2:end-1,2:end-2))
colorbar

subplot(3,2,6)
e2 = (Z2-Z2e);
surf(X2(2:end-1,2:end-2),Y2(2:end-1,2:end-2),e2(2:end-1,2:end-2))
colorbar

