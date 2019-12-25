function [U,rsd] = Jacobi_Poisson( N, h, U, F, k, omega)
%
% Jacobi iterations for the discretization of
%       - d2u/dx2 - d2u/dy2 = f(x,y)
%

  ind = 2:N-1;
  
  Un = U;
  
  for i = 1:k
    tmp = ( Un(ind-1,ind) + Un(ind+1,ind) + Un(ind,ind-1) + Un(ind,ind+1) )/4 + F(ind,ind)*(h^2)/4;
    U(ind,ind) = omega*tmp + (1-omega)*U(ind,ind);
    Un = U;
  end

  rsd = zeros(N,N);
  rsd(ind,ind) =  F(ind,ind) + ( U(ind-1,ind) + U(ind+1,ind) - 4*U(ind,ind)  + U(ind,ind-1) + U(ind,ind+1) ) / (h^2);
  
end