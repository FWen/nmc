clear all; clc; % close all;

WIDTH = 512;
X = imresize(double(imread('boat.png')),[WIDTH, WIDTH]);
[m,n] = size(X);

[S,V,D] = svd(X);
v = (diag(V));
% figure(2);
% plot(1:n,v/max(v),'-'); xlim([1,n]);

figure(3);subplot(121);
semilogy(1:m,v,'r-','linewidth',1); xlim([1,512]);%ylim([0.1,1e5]);
ylabel('Singular value');title('No-strictly low-rank');
legend('No-strictly low-rank');

% ---strictly low-rank----------
r = round(0.15*WIDTH);
[S,V,D] = svd(X);
X = S(:,1:r)*V(1:r,1:r)*D(:,1:r)';
% figure(1);subplot(2,2,2);
% imshow(uint8(X2));

subplot(122);
semilogy(1:m,v,'b-','linewidth',2); xlim([1,512]);%ylim([0.1,1e5]);
ylabel('RelErr');xlabel('q');
legend('SLR');title('Strictly low-rank')


J = randperm(m*n); 
J = J(1:round(0.5*m*n));    % sampling ratio
P = zeros(m*n,1);
P(J) = 1;
P = reshape(P,[m,n]);   % Projection Matrix

%----entry-wise noise----------
Y = X(:);
SNR = 40;
noise = randn(m*n,1);
noise = noise/std(noise) *10^(-SNR/20)*std(Y);  
Y = Y + noise;

% ---partial observation---
Y = reshape(Y,[m,n]).*P; 


figure(1); subplot(1,3,1);
imshow(uint8(Y));
title('Partial observation (50%)');


%  L1 --------------------
lamdas = logspace(-1,7,40);
parfor k = 1:length(lamdas); 
%     k=15
    [Xr, out]   = lq_pgd_mc(1, Y, P, lamdas(k), X, zeros(size(Y)));
    relerr1(k)  = norm(Xr-X,'fro')/norm(X,'fro');
    PSNR1(k) = psnr(X, Xr);
    X_l1(:,:,k) = Xr;
    
%   figure();semilogy(1:length(out.e),out.e,'-');
end
% figure(4);semilogy(lamdas,relerr1,'r-o');set(gca,'xscale','log');

[RelErr1, mi] = min(relerr1);
X_L1 = X_l1(:,:,mi); 

figure(1); subplot(1,3,2);
imshow(uint8(X_L1));
title(sprintf('Soft (RelErr=%.5f, PSNR=%.2f dB)', RelErr1, PSNR1(mi)));

%  Lq --------------------
qs = 0:0.1:1;
for kq=1:length(qs)
    qs(kq)
    parfor k=1:length(lamdas)       
        [Xr,~]       = lq_pgd_mc(qs(kq), Y, P, lamdas(k), X, X_L1);
        relerr(kq,k) = norm(Xr-X,'fro')/norm(X,'fro');
        PSNR(kq,k)   = psnr(X, Xr);
        X_lq(:,:,k)  = Xr;
    end
%     figure();semilogy(lamdas,relerr(kq,:),'r-o');set(gca,'xscale','log');
    
    [RelErr, mi]  = min(relerr(kq,:));
    Imgrs(:,:,kq) = X_lq(:,:,mi);
    RelErrs(kq)   = RelErr;
    PSNRr(kq)     = PSNR(kq,mi);
end

v0 = min(min(PSNR));
v1 = max(max(PSNR));
figure(6);
contourf(qs,lamdas,PSNR',[v0:0.5:v1]); colorbar; ylabel('\lambda'); xlabel('q');
set(gca, 'CLim', [v0, v1]);set(gca,'yscale','log');

[w1, e1] = max(PSNR'); [~, lo] = max(w1); ko = e1(lo);
figure(6);hold on;
plot(qs(lo),lamdas(ko),'r*');hold off;

[min_RelErr, mi] = min(RelErrs);

figure(1);subplot(1,3,3);
imshow(uint8(Imgrs(:,:,mi)));
title(['Lq (best q=', num2str(qs(mi), '%.1f'), ', ', 'RelErr=', num2str(min_RelErr, '%.5f'),', ', 'PSNR=', num2str(PSNRr(mi), '%.2f') ' dB)']);

