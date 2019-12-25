function X1 = restrict( X0, N1 )

    X1 = zeros(N1,N1);
    
    ind = 2:N1-1;
    tmp = 1*( X0(2*ind-2,2*ind-2) + X0(2*ind-2,2*ind  ) + X0(2*ind  ,2*ind-2) + X0(2*ind  ,2*ind) ) + ...
          2*( X0(2*ind-2,2*ind-1) + X0(2*ind  ,2*ind-1) + X0(2*ind-1,2*ind-2) + X0(2*ind-1,2*ind) ) + ... 
          4*( X0(2*ind-1,2*ind-1  ) );
    X1(ind,ind) = tmp/16;

end