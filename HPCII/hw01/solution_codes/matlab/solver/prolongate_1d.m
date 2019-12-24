function X0 = prolongate_1d(X1, N1, N0)
  
  X0 = zeros(N0,1);
  
  ind = 2:N1-1;
  X0(2*ind-1) = X1(ind);
  
  ind1 = 2:N1; ind2 = 1:N1-1;
  X0( 2*ind2 ) = 0.5 *( X1(ind1-1) + X1(ind2+1) );
  
end