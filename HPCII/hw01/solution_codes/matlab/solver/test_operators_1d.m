clc; clear

% f = @(x) x;
f = @(x) x.*(1-x);

k1 = 10;
k2 = 9;

N1 = 2^k1+1;
x1 = linspace(0,1,N1)';
z1e = f(x1);



N2 = 2^k2+1;
x2 = linspace(0,1,N2)';
z2e = f(x2);

z1 = prolongate_1d( z2e, N2, N1);
z2 = restrict_1d( z1e, N2 );



subplot(2,2,1)
plot(x1(2:end-1),z1e(2:end-1),'LineWidth',2); hold on
plot(x1(2:end-1),z1(2:end-1),'LineWidth',2)
grid on

subplot(2,2,2)
plot(x2(2:end-1),z2e(2:end-1),'LineWidth',2); hold on
plot(x2(2:end-1),z2(2:end-1),'LineWidth',2)
grid on

subplot(2,2,3)
e1 = (z1-z1e);
plot(x1(2:end-1),e1(2:end-1),'LineWidth',2)
grid on

subplot(2,2,4)
e2 = (z2-z2e);
plot(x2(2:end-1),e2(2:end-1),'LineWidth',2)
grid on

