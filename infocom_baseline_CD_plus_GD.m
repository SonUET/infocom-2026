clear; clc; close all;
%% ================== System & geometry ==================
c  = 300;           % [m/us]
lambda = 1;
d = lambda/2;
Rs = 100;           % [MHz]
Ts = 1/Rs;          % [us]
N  = 20;            % #subcarriers
Nt = 32;            % BS Tx ULA elements
Nr = Nt;            % BS Rx ULA elements
M  = 32;            % RIS ULA elements
Ns = M;
sigma = 0;          % noise std

% Geometry
posB = [0; 0];
posR = [5; 5];
posT = [10; 3];

% True angles
alpha   = atan2(posT(2)-posB(2), posT(1)-posB(1));   % unknown
thetaBR = atan2(posR(2)-posB(2), posR(1)-posB(1));   % known
thetaRB = atan2(posB(2)-posR(2), posB(1)-posR(1));   % known
beta    = atan2(posT(2)-posR(2), posT(1)-posR(1));   % unknown

% True distances / delays (echo)
d_BT_true  = norm(posT-posB);
d_BR_true  = norm(posR-posB);
d_RT_true  = norm(posT-posR);

tau_dir = 2*d_BT_true/c;
tau_ris = 2*(d_BR_true + d_RT_true)/c;

% Gains (complex)
g_dir_true = 9*exp(1j*pi/10);
g_ris_true = 6*exp(-1j*pi/8);

% Subcarrier angular freq
omega = 2*pi*((0:N-1)/(N*Ts));  % [rad/us]

%% ================== ULA steering ==================
aULA  = @(N_ant,th) exp(-1j*pi*(0:N_ant-1).'*sin(th))/sqrt(N_ant);
daULA = @(N_ant,th) (-1j*pi*cos(th)) * ((0:N_ant-1).'.*aULA(N_ant,th));

u_tx  = @(th) sqrt(Nt)*aULA(Nt,th);
u_rx  = @(th) sqrt(Nr)*aULA(Nr,th);

aRIS  = @(th) aULA(M, th);
daRIS = @(th) daULA(M, th);

a_RB = aRIS(thetaRB);

%% ================== RIS phases ==================
Phi = cell(N, Ns);
for n = 1:N
    for k = 1:Ns
        phi_nk   = 2*pi*rand(M,1);
        Phi{n,k} = diag(exp(1j*phi_nk));
    end
end

%% ================== Generate measurements y ==================
y = zeros(Nr, Ns, N);
F = zeros(Nt, Ns, N);

for n = 1:N
    ph_dir = exp(-1j*omega(n)*tau_dir);
    ph_ris = exp(-1j*omega(n)*tau_ris);

    for k = 1:Ns
        F(:,k,n) = exp(1j*2*pi*rand(Nt,1));  % random beams

        % -------- Direct echo --------
        y(:,k,n) = y(:,k,n) + u_rx(alpha+pi) * (g_dir_true*ph_dir) * (u_tx(alpha)'*F(:,k,n));

        % -------- RIS echo --------
        s_tx_BR = (u_tx(thetaBR))' * F(:,k,n);
        g_nk    = (a_RB') * (Phi{n,k} * aRIS(beta));
        b_true  = g_nk^2;

        y(:,k,n) = y(:,k,n) + u_rx(thetaRB) * (g_ris_true*ph_ris) * s_tx_BR * b_true;

        % Noise
        if sigma > 0
            y(:,k,n) = y(:,k,n) + sigma/sqrt(2)*(randn(Nr,1)+1j*randn(Nr,1));
        end
    end
end

y_vec = stack3(y);

%% ================== CD + GD ==================
% Init position
par.posT = posT + [0.3; -0.25].*randn(2,1);

% Init gains
par.dir.g = g_dir_true * exp(1j*0.1*randn);
par.ris.g = g_ris_true * exp(1j*0.1*randn);

MaxIt   = 150;      % outer iterations
eta_pos = 1.5;      % damping for position GD

% Histories
hist.J_total = zeros(1,MaxIt);
hist.J_dir   = zeros(1,MaxIt);
hist.J_ris   = zeros(1,MaxIt);

hist.alpha   = zeros(1,MaxIt);
hist.beta    = zeros(1,MaxIt);
hist.tau_dir = zeros(1,MaxIt);
hist.tau_ris = zeros(1,MaxIt);

hist.xT      = zeros(1,MaxIt);
hist.yT      = zeros(1,MaxIt);

hist.gdir_abs= zeros(1,MaxIt);
hist.gris_abs= zeros(1,MaxIt);
tic
for it = 1:MaxIt
    % ===== derive geometry from current posT =====
    [alpha_c, beta_c, tau_dir_c, tau_ris_c, dBT_c, dRT_c] = geom_from_pos( ...
        par.posT, posB, posR, d_BR_true, c);

    % ===== build atoms & jacobians w.r.t angle/tau =====
    [phi_dir, dphi_dir_tau, dphi_dir_alpha] = build_phi_dir_and_jac( ...
        alpha_c, tau_dir_c, F, u_tx, u_rx, daULA, omega);

    [phi_ris, dphi_ris_tau, dphi_ris_beta] = build_phi_ris_and_jac( ...
        beta_c, tau_ris_c, F, u_tx, u_rx, Phi, a_RB, aRIS, daRIS, ...
        omega, thetaBR, thetaRB);

    %% ===== CD on gains (LS closed-form) =====
    % Direct gain with current RIS fixed
    rdir = y_vec - phi_ris * par.ris.g;
    par.dir.g = (phi_dir' * rdir) / (phi_dir' * phi_dir);
    edir = rdir - phi_dir * par.dir.g;

    % RIS gain with current DIRECT fixed
    rris = y_vec - phi_dir * par.dir.g;
    par.ris.g = (phi_ris' * rris) / (phi_ris' * phi_ris);
    eris = rris - phi_ris * par.ris.g;

    %% ===== Position GD using chain rule =====
    % Compute partials of angles and delays w.r.t x,y
    [dalpha_dx, dalpha_dy, dtauD_dx, dtauD_dy, dbeta_dx,  dbeta_dy,  dtauR_dx, dtauR_dy] = geom_partials(par.posT, posB, posR, c);

    % dphi/dx, dphi/dy for each path
    dphi_dir_dx = dphi_dir_alpha * dalpha_dx + dphi_dir_tau * dtauD_dx;
    dphi_dir_dy = dphi_dir_alpha * dalpha_dy + dphi_dir_tau * dtauD_dy;

    dphi_ris_dx = dphi_ris_beta  * dbeta_dx  + dphi_ris_tau * dtauR_dx;
    dphi_ris_dy = dphi_ris_beta  * dbeta_dy  + dphi_ris_tau * dtauR_dy;

    % Full residual
    e = y_vec - phi_dir*par.dir.g - phi_ris*par.ris.g;

    % Gradient of J w.r.t x and y
    gx = -2*real( (conj(par.dir.g)*(dphi_dir_dx') + conj(par.ris.g)*(dphi_ris_dx')) * e );
    gy = -2*real( (conj(par.dir.g)*(dphi_dir_dy') + conj(par.ris.g)*(dphi_ris_dy')) * e );

    % Normalized
    denom_x = 2*( abs(par.dir.g)^2 * real(dphi_dir_dx'*dphi_dir_dx) + abs(par.ris.g)^2 * real(dphi_ris_dx'*dphi_ris_dx));
    denom_y = 2*( abs(par.dir.g)^2 * real(dphi_dir_dy'*dphi_dir_dy) + abs(par.ris.g)^2 * real(dphi_ris_dy'*dphi_ris_dy));

    dx = - gx / denom_x;
    dy = - gy / denom_y;

    par.posT = par.posT + eta_pos * [dx; dy];

    % ===== Log =====
    [alpha_c, beta_c, tau_dir_c, tau_ris_c, ~, ~] = geom_from_pos(par.posT, posB, posR, d_BR_true, c);

    % rebuild atoms for total cost logging
    [phi_dir, ~, ~] = build_phi_dir_and_jac(alpha_c, tau_dir_c, F, u_tx, u_rx, daULA, omega);
    [phi_ris, ~, ~] = build_phi_ris_and_jac(beta_c, tau_ris_c, F, u_tx, u_rx, Phi, a_RB, aRIS, daRIS, omega, thetaBR, thetaRB);
    yhat = phi_dir*par.dir.g + phi_ris*par.ris.g;

    hist.J_total(it) = norm(y_vec - yhat)^2;
    hist.J_dir(it)   = norm(edir)^2;
    hist.J_ris(it)   = norm(eris)^2;

    hist.alpha(it)   = alpha_c;
    hist.beta(it)    = beta_c;
    hist.tau_dir(it) = tau_dir_c;
    hist.tau_ris(it) = tau_ris_c;

    hist.xT(it)      = par.posT(1);
    hist.yT(it)      = par.posT(2);

    hist.gdir_abs(it)= abs(par.dir.g);
    hist.gris_abs(it)= abs(par.ris.g);

    fprintf('It%3d | J=%.3e | xT=%.3f yT=%.3f | alpha=%.2f° beta=%.2f° | tauD=%.4f tauR=%.4f\n',...
        it, hist.J_total(it), par.posT(1), par.posT(2), ...
        rad2deg(alpha_c), rad2deg(beta_c), tau_dir_c, tau_ris_c);
end
toc

% save('geom_cd_gd_history.mat','hist','par');

%% ================== Plots ==================
figure;
subplot(2,2,1);
semilogy(hist.J_total,'-o','LineWidth',1.5); grid on;
title('Total cost $\|y-\phi_d g_d-\phi_r g_r\|^2$','Interpreter','latex');
xlabel('Iteration'); ylabel('Cost');

subplot(2,2,2);
semilogy(hist.J_dir,'-o'); hold on; semilogy(hist.J_ris,'-o'); grid on;
legend('J\_dir','J\_ris','Location','best');
title('Conditional costs'); xlabel('Iteration');

subplot(2,2,3);
plot(rad2deg(hist.alpha),'-o'); hold on; plot(rad2deg(hist.beta),'-o'); grid on;
legend('\alpha','\beta','Location','best');
title('Angles (deg)'); xlabel('Iteration'); ylabel('deg');

subplot(2,2,4);
plot(hist.tau_dir,'-o'); hold on; plot(hist.tau_ris,'-o'); grid on;
legend('\tau\_dir','\tau\_ris','Location','best');
title('Delays (us)'); xlabel('Iteration'); ylabel('us');

figure;
subplot(1,2,1);
plot(hist.xT,'-o'); hold on; yline(posT(1),'--'); grid on;
xlabel('Iteration'); ylabel('x'); title('Target x-coordinate');
legend('est','true');

subplot(1,2,2);
plot(hist.yT,'-o'); hold on; yline(posT(2),'--'); grid on;
xlabel('Iteration'); ylabel('y'); title('Target y-coordinate');
legend('est','true');

figure;
plot(hist.gdir_abs,'-o','LineWidth',1.2); hold on;
plot(hist.gris_abs,'-o','LineWidth',1.2); grid on;
legend('|g\_dir|','|g\_ris|','Location','best');
xlabel('Iteration');
save('cdgd.mat', '-struct', 'hist', 'gdir_abs', 'gris_abs');

function v = stack3(Y)
    [Nr,Ns,N] = size(Y);
    v = zeros(Nr*Ns*N,1);
    for n=1:N
        v((n-1)*Nr*Ns + (1:Nr*Ns)) = reshape(Y(:,:,n), Nr*Ns, 1);
    end
end

function [alpha, beta, tauD, tauR, dBT, dRT] = geom_from_pos(posT, posB, posR, dBR, c)
    dB = posT - posB;
    dR = posT - posR;

    dBT = norm(dB);
    dRT = norm(dR);

    alpha = atan2(dB(2), dB(1));
    beta  = atan2(dR(2), dR(1));

    tauD = 2*dBT/c;
    tauR = 2*(dBR + dRT)/c;
end

function [dalpha_dx, dalpha_dy, dtauD_dx, dtauD_dy, dbeta_dx,  dbeta_dy,  dtauR_dx, dtauR_dy] = geom_partials(posT, posB, posR, c)

    dB = posT - posB;  xB = dB(1); yB = dB(2);
    dR = posT - posR;  xR = dR(1); yR = dR(2);

    rB = norm(dB);
    rR = norm(dR);

    % alpha = atan2(yB,xB)
    dalpha_dx = -yB/(rB^2);
    dalpha_dy =  xB/(rB^2);

    % beta = atan2(yR,xR)
    dbeta_dx  = -yR/(rR^2);
    dbeta_dy  =  xR/(rR^2);

    % tau_dir = 2*rB/c
    dtauD_dx  = (2/c) * (xB/rB);
    dtauD_dy  = (2/c) * (yB/rB);

    % tau_ris = 2(dBR + rR)/c  => derivative depends only on rR
    dtauR_dx  = (2/c) * (xR/rR);
    dtauR_dy  = (2/c) * (yR/rR);
end

function [phi, dphi_tau, dphi_alpha] = build_phi_dir_and_jac(alpha, tau, F, u_tx, u_rx, daULA, omega)

    [Nt, Ns, N] = size(F);
    Nr = Nt;

    phi        = zeros(Nr*Ns*N,1);
    dphi_tau   = zeros(size(phi));
    dphi_alpha = zeros(size(phi));

    % unit-norm steerings
    a_tx  = exp(-1j*pi*(0:Nt-1).'*sin(alpha))/sqrt(Nt);
    a_rx  = exp(-1j*pi*(0:Nr-1).'*sin(alpha+pi))/sqrt(Nr);

    da_tx = daULA(Nt, alpha);
    da_rx = daULA(Nr, alpha+pi);

    utx  = u_tx(alpha);
    urx  = u_rx(alpha+pi);

    dutx = sqrt(Nt)*da_tx;
    durx = sqrt(Nr)*da_rx;

    base = 0;
    for n=1:N
        ph = exp(-1j*omega(n)*tau);
        for k=1:Ns
            s  = (utx'  * F(:,k,n));
            ds = (dutx' * F(:,k,n));

            atom = urx * (ph*s);
            dA   = durx * (ph*s) + urx * (ph*ds);

            idx = base + (k-1)*Nr + (1:Nr);
            phi(idx)        = atom;
            dphi_tau(idx)   = (-1j*omega(n)) * atom;
            dphi_alpha(idx) = dA;
        end
        base = base + Nr*Ns;
    end
end

function [phi, dphi_tau, dphi_beta] = build_phi_ris_and_jac(beta, tau, F, u_tx, u_rx, Phi, a_RB, aRIS, daRIS, omega, thetaBR, thetaRB)

    [Nt, Ns, N] = size(F);
    Nr = Nt;

    phi       = zeros(Nr*Ns*N,1);
    dphi_tau  = zeros(size(phi));
    dphi_beta = zeros(size(phi));

    urx   = u_rx(thetaRB);
    utxBR = u_tx(thetaBR);

    aRT  = aRIS(beta);
    daRT = daRIS(beta);

    base = 0;
    for n=1:N
        ph   = exp(-1j*omega(n)*tau);
        srow = (utxBR' * F(:,:,n));

        for k=1:Ns
            s = srow(k);

            g_nk  = (a_RB') * (Phi{n,k} * aRT);
            dg_nk = (a_RB') * (Phi{n,k} * daRT);

            b_nk  = g_nk^2;
            db_nk = 2 * g_nk * dg_nk;

            atom = urx * (ph*s*b_nk);
            dB   = urx * (ph*s*db_nk);

            idx = base + (k-1)*Nr + (1:Nr);
            phi(idx)       = atom;
            dphi_tau(idx)  = (-1j*omega(n)) * atom;
            dphi_beta(idx) = dB;
        end

        base = base + Nr*Ns;
    end
end
