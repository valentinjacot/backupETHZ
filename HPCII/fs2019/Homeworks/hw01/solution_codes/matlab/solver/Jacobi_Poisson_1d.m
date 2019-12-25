function [U,rsd] = Jacobi_Poisson_1d( N, h, U, F, k, omega)
%
% Jacobi iterations for the discretization of
%       - d2u/dx2 = f(x,y)
%

  ind = 2:N-1;
  
  Un = U;
   
  for i = 1:k
    tmp = ( Un(ind-1) + Un(ind+1) )/2 + F(ind)*(h^2)/2;
    U(ind) = omega*tmp + (1-omega)*U(ind);
    Un = U;
  end

  rsd = zeros(N,1);
  rsd(ind) =  F(ind) + ( U(ind-1) - 2*U(ind) + U(ind+1)  ) / (h^2);
  
  
end