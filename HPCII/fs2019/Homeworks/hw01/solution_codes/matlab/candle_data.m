%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to generate data for the Candle Problem 
% 
% (Iterative Multigrid Jacobi Solver for the 2D Poisson eq.)  
%       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('./solver')

clear; clc
close all

%% Data options

write = true; % write to file candle_data.txt?

c1 = 10;    % intensity
c2 = 0.01;  % variance
c3 = 0.5;   % x0
c4 = 0.5;   % y0

points = [ 0.2, 0.4, 0.5, 0.2, 0.7, 0.8;
            0.4, 0.4, 0.3, 0.4, 0.6, 0.8];
        
err_sigma = 0.1;

%% Choose which problem to solve
pde.bc        =   @ pde_bc_candle;
pde.rhs       =   @ (N) pde_rhs_candle(N, c1, c2, c3, c4);
pde.solution  =   @ zeros;

%% Initializing solver variables
sp.tolerance     = 1e-6;     % difference in residual Tolerance
sp.maxIter       = 1e4;      % maximum number of multigrid iterations
sp.UsePlotting   = false;    % use false to disable plotting and measure performance properly.

sp.omega = 2/3;

sp.k1 = 3;   % number of relaxation iterations going down
sp.k2 = 3;   % number of relaxation iterations going up

sp.L = [ 5 4 ];

N = 2^sp.L(1)+1;
sp.U0 = zeros( N, N );

%% Solve
[ U, rsd ] = multigrid_poisson( pde, sp );

%% Generate Data
[ D, D_true ] = generate_data(U, points, err_sigma);


%% Write to file
if (write)
  out = [points; D];
  fileID = fopen('candle_data.txt','w');
  fprintf(fileID,'%f %f %f\n', out);
  fclose(fileID);
end

rmpath('./solver')

%% Problem: non-homogeneous with zero bc (candle)

function U = pde_bc_candle( U )
  U(1,:) = 0;  U(end,:) = 0;
  U(:,1) = 0;  U(:,end) = 0;
end


function F = pde_rhs_candle( N, c1, c2, c3, c4 )
% c1: intensity
% c2: variance
% c3: x0
% c4: y0

  x = linspace(0,1,N);
  [X,Y]=meshgrid(x,x);
    
  % Using this F the solution will be a gaussian:
  %   U = c1*exp( - ( (X-c3).^2 + (Y-c4).^2 )/c2 );
  F = - (4*c1*exp(-((c3 - X).^2 + (c4 - Y).^2)/c2).*(c3.^2 - 2*c3*X + c4^2 - 2*c4*Y + X.^2 + Y.^2 - c2))/c2^2;
  
end


%% Generate Data Function

function [ D, D_true ] = generate_data( U, xy, sigma)

  % linearly interpolate U through points xy
  % add Gaussian error with sdev sigma
  
  dx = linspace(0,1,size(U{1},1));
  dy = linspace(0,1,size(U{1},2));
      
  D_true = interp2( dx, dy, U{1}, xy(1,:), xy(2,:));
  D = D_true + sigma * randn(1,length(xy));

end