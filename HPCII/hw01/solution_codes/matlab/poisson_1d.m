%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iterative Multigrid Jacobi Solver for the 2D Poisson eq.  
%       
%     - d2u/dx2 = f(x,y),   x in [0,1]                             
% 
%         subject to Dirichlet boundary conditions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



addpath('./solver')

clear; clc
close all

%% Choose which problem to solve
pde.bc        =   @ pde_bc_p2;
pde.rhs       =   @ pde_rhs_p2;
pde.solution  =   @ pde_solution_p2;


%% Initializing solver variables
sp.tolerance     = 1e-6;     % difference in residual Tolerance
sp.maxIter       = 1e4;      % maximum number of multigrid iterations
sp.UsePlotting   = true;     % use false to disable plotting and measure performance properly.

sp.omega = 2/3;

sp.k1 = 3;   % number of relaxation iterations going down
sp.k2 = 3;   % number of relaxation iterations going up

sp.L = [ 8 7 6 5 4  ];

N = 2^sp.L(1)+1;

sp.U0 =  1+zeros( N, 1 );


%% Solve
[ U, rsd ] = multigrid_poisson_1d( pde, sp );


rmpath('./solver')




%% Problem 1: Homogeneous with zero bc

function U = pde_bc_p1( U )
  U(1) = 0;  U(end) = 0;
end

function F = pde_rhs_p1( N )
  F = zeros( N, 1 );
end

function F = pde_solution_p1( N )
  F = zeros( N, 1 );
end



%% Problem 2: non-homogeneous with zero bc

function U = pde_bc_p2( U )
  U(1) = 0;  U(end) = 0;
end

function F = pde_rhs_p2( N )
  F = 2*ones(N,1);
end

function F = pde_solution_p2( N )
  x = linspace(0,1,N)';  
  F = x.*(1-x);
end
