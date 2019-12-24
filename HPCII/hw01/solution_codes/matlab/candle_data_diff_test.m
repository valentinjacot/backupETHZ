%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to generate data from two simulations with different
%   accuracy level (L1 'high'/ L2 'low')
% 
% Sample points from both solutions and compare
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('./solver')

%clear; clc
close all

write = true; % write diff to file candle_diff_data.txt?

%% Accuracy levels

L1 = [ 9 8 ];
L2 = [ 7 6 ];

%% Data options

NP = 2^7;                     
[x,y]  = meshgrid(0:1/NP:1);  
points = [x(:),y(:)]'; % interpolate solution for L1 on L2 grid points (below)
        

%% IC 'Problem' Parameters
c1 = 10;    % intensity
c2 = 0.01;  % variance
c3 = 0.5;   % x0
c4 = 0.5;   % y0

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


%% Solve
% accurate
sp.L  = L1;
N = 2^sp.L(1)+1;
sp.U0 = zeros( N, N );
%[ U1, ~ ] = multigrid_poisson( pde, sp );

% approx
sp.L = L2;
N = 2^sp.L(1)+1;
sp.U0 = zeros( N, N );
[ U2, ~ ] = multigrid_poisson( pde, sp );


%% Sample Data
[ ~ , D1_true ] = generate_data(U1, points, 0.0);
[ ~ , D2_true ] = generate_data(U2, points, 0.0);

% Compare
diff = D1_true - D2_true;
[M,I] = max(abs(diff))
L1err = sum(abs(diff))/length(diff)
L2err = sum(diff.^2)/length(diff)

if (write)
  out = [points; D1_true; D2_true; diff];
  fileID = fopen('candle_diff_data.txt','w');
  fprintf(fileID,'%f %f %f %f %f\n', out);
  fclose(fileID);
end

rmpath('./solver')

%% Problem: non-homogeneous with zero bc (candle)

function U = pde_bc_candle( U )
  U(1,:) = 0;  U(end,:) = 0;
  U(:,1) = 0;  U(:,end) = 0;
end


function F = pde_rhs_candle( N, c1, c2, c3, c4 )

  x = linspace(0,1,N);
  [X,Y]=meshgrid(x,x);
    
  % Using this F the solution will be a gaussian:
  %   U = c1*exp( - ( (X-c3).^2 + (Y-c4).^2 )/c2 );
  F = - (4*c1*exp(-((c3 - X).^2 + (c4 - Y).^2)/c2).*(c3.^2 - 2*c3*X + c4^2 - 2*c4*Y + X.^2 + Y.^2 - c2))/c2^2;
  
end


%% Generate Data Function

function [ D, D_true ] = generate_data( U, xy, sigma)

  % linearly interpolate U{1} through points xy
  % add Gaussian error with sdev sigma
  % (can be used to compare U's)
  
  dx = linspace(0,1,size(U{1},1));
  dy = linspace(0,1,size(U{1},2));
      
  D_true = interp2( dx, dy, U{1}, xy(1,:), xy(2,:));
  D = D_true + sigma * randn(1,length(xy));

end