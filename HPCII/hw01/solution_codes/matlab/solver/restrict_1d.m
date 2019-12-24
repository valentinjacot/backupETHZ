function X1 = restrict_1d( X0, N1 )

X1 = zeros(N1,1);

ind = 2:N1-1;

tmp = X0(2*ind-2) + ...
      X0(2*ind) + ... 
      2*X0(2*ind-1) ;

X1(ind) = tmp/4;


% X1(ind) = X0(2*ind-1) ;

end