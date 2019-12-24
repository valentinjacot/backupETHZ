function X0 = prolongate(X1, N1, N0)
  
  X0 = zeros(N0,N0);
  
  ind = 2:N1-1;
  X0(2*ind-1,2*ind-1) = X1(ind,ind);
  
  ind1 = 2:N1; ind2 = 2:N1-1;
  X0(2*ind1-2,2*ind2-1) = 0.5 *( X1(ind1-1,ind2) + X1(ind1,ind2) );

  ind1 = 2:N1-1; ind2 = 2:N1;
  X0(2*ind1-1,2*ind2-2) = 0.5 *( X1(ind1,ind2-1) + X1(ind1,ind2) );

  ind1 = 2:N1; ind2 = 2:N1;
  X0(2*ind1-2,2*ind2-2) = 0.25*( X1(1:N1-1,1:N1-1) + X1(1:N1-1,2:N1) + X1(2:N1,1:N1-1) + X1(2:N1,2:N1) );
  
end