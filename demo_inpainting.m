clear all; clc; close all;

WIDTH = 512;
X = imresize(double(imread('boat.png')),[WIDTH, WIDTH]);
[m,n] = size(X);

[S,V,D] = svd(X);
v = (diag(V));

figure(1);subplot(121);
semilogy(1:m,v,'r-','linewidth',1); xlim([1,512]);%ylim([0.1,1e5]);
ylabel('Singular value');title('No-strictly low-rank');
legend('No-strictly low-rank');

% ---strictly low-rank----------
r = round(0.15*WIDTH);
[S,V,D] = svd(X);
X = S(:,1:r)*V(1:r,1:r)*D(:,1:r)';


subplot(122);
semilogy(1:m,v,'b-','linewidth',2); xlim([1,512]);%ylim([0.1,1e5]);
ylabel('RelErr');xlabel('q');
legend('SLR');title('Strictly low-rank')


J = randperm(m*n); 
J = J(1:round(0.5*m*n));    % sampling ratio
P = zeros(m*n,1);
P(J) = 1;
P = reshape(P,[m,n]);   % Projection Matrix

% ----entry-wise noise----------
Y = X(:);
SNR = 40;
noise = randn(m*n,1);
noise = noise/std(noise) *10^(-SNR/20)*std(Y);  
Y = Y + noise;

% ---partial observation---------
Y = reshape(Y,[m,n]).*P; 


figure(2); subplot(1,3,1);
imshow(uint8(Y));
title('Partial observation (50%)');


%  L1 --------------------
[X_l1, out]  = lq_pgd_mc(1, Y, P, 10, X, zeros(size(Y)));
relerr = norm(X_l1-X,'fro')/norm(X,'fro');

figure(1); subplot(1,3,2);
imshow(uint8(X_l1));
title(sprintf('Soft-PGD (RelErr=%.5f, PSNR=%.2f dB)', relerr, psnr(X, X_l1)));


%  Lq --------------------
[X_lq,~]    = lq_pgd_mc(0.1, Y, P, 1e3, X, X_l1);
relerr = norm(X_lq-X,'fro')/norm(X,'fro');

figure(2);subplot(1,3,3);
imshow(uint8(X_lq));
title(sprintf('Lq-PGD (RelErr=%.5f, PSNR=%.2f dB)', relerr, psnr(X, X_lq)));
