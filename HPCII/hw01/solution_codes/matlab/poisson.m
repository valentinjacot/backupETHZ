%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iterative Multigrid Jacobi Solver for the 2D Poisson eq.  
%       
%     - d2u/dx2 - d2u/dy2 = f(x,y),   (x,y) in [0,1]x[0,1]                              
% 
%         subject to Dirichlet boundary conditions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



addpath('./solver')

clear; clc
close all

%% Choose which problem to solve
pde.bc        =   @ pde_bc_p4;
pde.rhs       =   @ pde_rhs_p4;
pde.solution  =  []; % @ pde_solution_p3;


%% Initializing solver variables
sp.tolerance     = 1e-6;     % difference in residual Tolerance
sp.maxIter       = 1e4;      % maximum number of multigrid iterations
sp.UsePlotting   = false;     % use false to disable plotting and measure performance properly.

sp.omega = 1;

sp.k1 = 3;   % number of relaxation iterations going down
sp.k2 = 3;   % number of relaxation iterations going up

sp.L = [ 7 6 5 4 3 2 ];

N = 2^sp.L(1)+1;
sp.U0 = zeros( N, N );


%% Solve
[ U, rsd ] = multigrid_poisson( pde, sp );


rmpath('./solver')



%% Problem 1: Homogeneous with zero bc

function U = pde_bc_p1( U )
  U(1,:) = 1;  U(end,:) = 1;
  U(:,1) = 1;  U(:,end) = 1;
end

function F = pde_rhs_p1( N )
  F = zeros( N, N );
end

function F = pde_solution_p1( N )
  F = 1 + zeros( N, N );
end



%% Problem 2: non-homogeneous with zero bc

function U = pde_bc_p2( U )
  U(1,:) = 0;  U(end,:) = 0;
  U(:,1) = 0;  U(:,end) = 0;
end

function F = pde_rhs_p2( N )
  x = linspace(0,1,N);
  [X,Y]=meshgrid(x,x);
  
  F = (1-6*X.^2).*(Y.^2).*(1-Y.^2) + (1-6*Y.^2).*(X.^2).*(1-X.^2);
  F = -2*F;
end

function F = pde_solution_p2( N )
  x = linspace(0,1,N);
  [X,Y]=meshgrid(x,x);  
  F = -(X.^2-X.^4).*(Y.^4-Y.^2);
end


%% Problem 3: non-homogeneous with zero bc (candle)

function U = pde_bc_p3( U )
  U(1,:) = 0;  U(end,:) = 0;
  U(:,1) = 0;  U(:,end) = 0;
end


function F = pde_rhs_p3( N )
  x = linspace(0,1,N);
  [X,Y]=meshgrid(x,x);
  
  c1 = 10; % intensity
  c2 = 0.01; % variance
  c3 = 0.5; % x0
  c4 = 0.5; % y0
  
  
  % Using this F the solution will be a gaussian:
  %   U = c1*exp( - ( (X-c3).^2 + (Y-c4).^2 )/c2 );
  F = - (4*c1*exp(-((c3 - X).^2 + (c4 - Y).^2)/c2).*(c3.^2 - 2*c3*X + c4^2 - 2*c4*Y + X.^2 + Y.^2 - c2))/c2^2;
  
end



%% Problem 4: non-homogeneous with zero bc (3 candles)

function U = pde_bc_p4( U )
  U(1,:) = 0;  U(end,:) = 0;
  U(:,1) = 0;  U(:,end) = 0;
end


function F = pde_rhs_p4( N )
  x = linspace(0,1,N);
  [X,Y]=meshgrid(x,x);
  
  c1 = 10; % intensity
  c2 = 0.01; % variance
  c3 = 0.3; % x0
  c4 = 0.3; % y0
  
  F1 = - (4*c1*exp(-((c3 - X).^2 + (c4 - Y).^2)/c2).*(c3.^2 - 2*c3*X + c4^2 - 2*c4*Y + X.^2 + Y.^2 - c2))/c2^2;
  
  c1 = 10; % intensity
  c2 = 0.01; % variance
  c3 = 0.7; % x0
  c4 = 0.3; % y0
  
  F2 = - (4*c1*exp(-((c3 - X).^2 + (c4 - Y).^2)/c2).*(c3.^2 - 2*c3*X + c4^2 - 2*c4*Y + X.^2 + Y.^2 - c2))/c2^2;
  
  
  c1 = 10; % intensity
  c2 = 0.01; % variance
  c3 = 0.5; % x0
  c4 = 0.7; % y0
  
  F3 = - (4*c1*exp(-((c3 - X).^2 + (c4 - Y).^2)/c2).*(c3.^2 - 2*c3*X + c4^2 - 2*c4*Y + X.^2 + Y.^2 - c2))/c2^2;
  
  
  F = F1+F2+F3;
  
end





