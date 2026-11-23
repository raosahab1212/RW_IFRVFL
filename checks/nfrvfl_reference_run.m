% Runs the authors' NF-RVFL training function on fixed inputs and saves the results.
% Needs Neuro_Fuzzy_RVFL_train.m and credit_approval.mat from github.com/mtanveer1/NeuroFuzzy-RVFL,
% copied as nf_train_given_centres.m with the centres passed in and the test scores returned (see README.md).
warning off all;
d = load('credit_approval.mat'); D = d.credit_approval;
X = D(:,1:end-1); y = D(:,end);
X = bsxfun(@rdivide, X - mean(X,1), std(X));
n = size(X,1); te0 = floor(0.8*n);
trX = X(1:te0-1,:); trY = y(1:te0-1); teX = X(te0:end,:); teY = y(te0:end);
rand('seed', 1); randn('seed', 1);
cases = [15 203 0.001; 20 700 0.1];
for c = 1:2
  N = cases(c,1); L = cases(c,2); C = cases(c,3);
  Alpha = rand(size(trX,2), N); W = rand(N+1, L); centres = trX(1:N,:);
  [~,~,~,acc,scores] = nf_train_given_centres(trX,trY,teX,teY,Alpha,W,C,N,5,2,3,centres);
  save('-mat7-binary', sprintf('case%d.mat', c), 'trX','trY','teX','teY','Alpha','W','centres','C','N','L','acc','scores');
  printf('case %d: N=%d L=%d C=%g test acc %.4f\n', c, N, L, C, acc);
end
