A = randn(10,8);
R = 4;
[offset,coor,delta,mu,phi] = rb_train(A,0,.5,R);
assert(numel(offset) == R+1 ...
       && all(size(coor) == [offset(end)-offset(1), size(A,2)]) ...
       && all(size(delta) == [size(A,2),R]) ...
       && all(size(mu) == [size(A,2),R]) ...
       && all(size(phi) == [size(A,1),size(coor,1)]))

phi0 = rb_test(sparse(A), offset, coor, delta, mu);
assert(norm(phi0 - phi, 1) < eps)

% Unit test

[offset, coor] = rb_train([eye(10); eye(10)], ones(10,1), zeros(10,1))
assert(all(offset == [1; 11]) && norm(eye(10) - double(coor)) < eps)
disp('Success')
