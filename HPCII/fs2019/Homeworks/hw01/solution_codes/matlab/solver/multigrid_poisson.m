function [ U, rsd ] = multigrid_poisson( pde, sp )
%
% pde: struct that defines the Poisson equation to be solved
% sp : struct with parameters for the solver
%

  nL = length(sp.L);

  N = 2.^sp.L + 1;  % Total number of points per side at the Finest Grid, including boundary cells.
                    % keep it power of two plus one.
  h = 1./(N-1);     % delta_x = delta_y



  U   = cell(nL,1);
  f   = cell(nL,1);
  rsd = cell(nL,1);

  
  U{1} = sp.U0;
  for i = 2:nL
    U{i} = zeros( N(i), N(i) );
  end


  % Setting initial residual and difference
  L2_rsd      = inf;
  L2_rsd_diff = inf;


  [ terr, trsd ] = deal( zeros(sp.maxIter,1) ); 
  
  % Set boundary conditions and rhs
  U{1} = pde.bc( U{1} );
  f{1} = pde.rhs( N(1) );

  if( isa(pde.solution,'function_handle') ) % not all problems will have an anaytic solution
    S = pde.solution( N(1) );
  else
    S = [];
  end



  %% Starting Solver - Iterate while residual is larger than tolerance
  iteration = 1;
  while ( L2_rsd_diff > sp.tolerance  &&  iteration < sp.maxIter )

    % relax k times
    [ U{1}, rsd{1} ] = Jacobi_Poisson( N(1), h(1), U{1}, f{1}, sp.k1, sp.omega);

    % go down
    for i = 2:nL

      % h -> 2h
      f{i} = restrict( rsd{i-1}, N(i) );

      % relax
      U{i} = zeros( N(i), N(i) );
      [ U{i}, rsd{i} ] = Jacobi_Poisson( N(i), h(i), U{i}, f{i}, sp.k1, sp.omega);

    end


    % go up
    for i = nL: -1 : 2

      % 2h -> h  and correct
      tmp = prolongate( U{i}, N(i), N(i-1) );
      U{i-1} = U{i-1} + tmp;

      % relax
      [ U{i-1}, rsd{i-1} ] = Jacobi_Poisson( N(i-1), h(i-1), U{i-1}, f{i-1}, sp.k2, sp.omega);

    end


    % Calculate new residual
    tmp         = L2_rsd;
    L2_rsd      = norm(rsd{1}(:));
    L2_rsd_diff = abs( L2_rsd - tmp );


    if (sp.UsePlotting)
      [terr, trsd] = plot_solution( h, U, rsd, S, terr, trsd, iteration  );
    end

    
    if isempty(S) 
      if mod(iteration,20)==0
        fprintf('          residual   |  residual dif.\n');
      end
      fprintf('%04d:   %e |  (%e) \n', iteration, L2_rsd, L2_rsd_diff  );
    else
      if mod(iteration,20)==0
        fprintf('          residual   |  residual dif. |    L2 error\n');
      end
      tmp = abs(U{1}-S);
      fprintf('%04d:   %e |  %e  |  %e  \n', iteration, L2_rsd, L2_rsd_diff, norm(tmp(:))  );
    end
    
    iteration = iteration + 1;

  end

  
  
  
end




%%

function [terr, trsd] = plot_solution( h, U, rsd, S, terr, trsd, n )
  
  fig = figure(1); clf

  ns = length(h);
  
  
  for i = 1:ns
  
    [X,Y] = meshgrid(0:h(i):1,0:h(i):1);
    
    subplot(ns+1,2,2*i-1)
    surf(X,Y,U{i},'FaceColor','interp')
    view(90,0);
    title([ 'Solution  L=' num2str(i) ])
    colorbar
    

    subplot(ns+1,2,2*i)
    surf(X,Y,rsd{i},'FaceColor','interp')
    view(90,0);
    title([ 'Residual  L=' num2str(i) ])
    colorbar
  
  end
  
  if ~isempty(S)
    tmp = abs(U{1}-S);
    terr(n) = norm(tmp(:));
  end
  
  
  trsd(n) = norm( rsd{1}(:) );

  
  if ~isempty(S)
  
    subplot(ns+1,2,2*ns+1)
    semilogy(1:n,terr(1:n), 'LineWidth',3)
    ylabel('L2 error')
    grid on
    
    subplot(ns+1,2,2*ns+2)
    semilogy(1:n,trsd(1:n), 'LineWidth',3)
    ylabel('L2 residual')
    grid on
  
  else
    
    subplot(ns+1,2, 2*ns+1:2*ns+2 )
    semilogy(1:n,trsd(1:n), 'LineWidth',3)
    ylabel('L2 residual')
    grid on
  
  end
  
  
  drawnow
  

end
