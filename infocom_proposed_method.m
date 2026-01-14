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
alpha_true   = atan2(posT(2)-posB(2), posT(1)-posB(1));   % unknown
thetaBR = atan2(posR(2)-posB(2), posR(1)-posB(1));        % known
thetaRB = atan2(posB(2)-posR(2), posB(1)-posR(1));        % known
beta_true    = atan2(posT(2)-posR(2), posT(1)-posR(1));   % unknown

% True distances / delays (echo)
d_BT_true  = norm(posT-posB);
d_BR_true  = norm(posR-posB);
d_RT_true  = norm(posT-posR);

tau_dir_true = 2*d_BT_true/c;
tau_ris_true = 2*(d_BR_true + d_RT_true)/c;

% Gains
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
    ph_dir = exp(-1j*omega(n)*tau_dir_true);
    ph_ris = exp(-1j*omega(n)*tau_ris_true);

    for k = 1:Ns
        F(:,k,n) = exp(1j*2*pi*rand(Nt,1));  % random beams

        % -------- Direct echo --------
        y(:,k,n) = y(:,k,n) + u_rx(alpha_true+pi) * (g_dir_true*ph_dir) * (u_tx(alpha_true)'*F(:,k,n));

        % -------- RIS echo --------
        s_tx_BR = (u_tx(thetaBR))' * F(:,k,n);
        g_nk    = (a_RB') * (Phi{n,k} * aRIS(beta_true));
        b_true  = g_nk^2;

        y(:,k,n) = y(:,k,n) + u_rx(thetaRB) * (g_ris_true*ph_ris) * s_tx_BR * b_true;

        % Noise
        if sigma > 0
            y(:,k,n) = y(:,k,n) + sigma/sqrt(2)*(randn(Nr,1)+1j*randn(Nr,1));
        end
    end
end

y_vec = stack3(y);

%% ================== LINEARIZED INNER-LOOP ==================
% Init position
par.posT = posT + [0.3; -0.25].*randn(2,1);

OuterIt = 15;               % outer loop
InnerIt = 10;               % inner loop
MaxSteps = OuterIt*InnerIt;

mu0   = 1e-3;
muUp  = 10;
muDn  = 0.3;
maxTrials = 6;

hist.J_total = zeros(1,MaxSteps);
hist.alpha   = zeros(1,MaxSteps);
hist.beta    = zeros(1,MaxSteps);
hist.tau_dir = zeros(1,MaxSteps);
hist.tau_ris = zeros(1,MaxSteps);
hist.xT      = zeros(1,MaxSteps);
hist.yT      = zeros(1,MaxSteps);
hist.gdir_abs_pp= zeros(1,MaxSteps);
hist.gris_abs_pp= zeros(1,MaxSteps);

step = 0;
mu = mu0;

tic
for out = 1:OuterIt
    % ===== Base geometry at start of outer loop =====
    [alpha0, beta0, tauD0, tauR0, ~, ~] = geom_from_pos(par.posT, posB, posR, d_BR_true, c);

    % ===== Build atoms & jacobians =====
    [phi_dir0, dphi_dir_tau0, dphi_dir_alpha0] = build_phi_dir_and_jac( ...
        alpha0, tauD0, F, u_tx, u_rx, daULA, omega);

    [phi_ris0, dphi_ris_tau0, dphi_ris_beta0] = build_phi_ris_and_jac( ...
        beta0, tauR0, F, u_tx, u_rx, Phi, a_RB, aRIS, daRIS, omega, thetaBR, thetaRB);

    for in = 1:InnerIt
        step = step + 1;

        % ===== Current geometry =====
        [alpha_c, beta_c, tauD_c, tauR_c, ~, ~] = geom_from_pos(par.posT, posB, posR, d_BR_true, c);

        % ===== Linearized atom =====
        dAlpha = wrapToPi(alpha_c - alpha0);
        dBeta  = wrapToPi(beta_c  - beta0);
        dTauD  = (tauD_c - tauD0);
        dTauR  = (tauR_c - tauR0);

        phi_dir = phi_dir0 + dphi_dir_alpha0*dAlpha + dphi_dir_tau0*dTauD;
        phi_ris = phi_ris0 + dphi_ris_beta0 *dBeta  + dphi_ris_tau0*dTauR;

        % ===== Joint LS for gains =====
        Phi2 = [phi_dir, phi_ris];
        g2   = (Phi2' * Phi2) \ (Phi2' * y_vec);
        gdir = g2(1);
        gris = g2(2);

        yhat = Phi2*g2;
        e    = y_vec - yhat;
        Jcur = real(e'*e);

        % ===== LM update for position using linearized derivatives =====
        [dalpha_dx, dalpha_dy, dtauD_dx, dtauD_dy, dbeta_dx, dbeta_dy, dtauR_dx, dtauR_dy] ...
            = geom_partials(par.posT, posB, posR, c);

        % Use base derivatives (dphi_*0) but current geom partials
        dphi_dir_dx = dphi_dir_alpha0 * dalpha_dx + dphi_dir_tau0 * dtauD_dx;
        dphi_dir_dy = dphi_dir_alpha0 * dalpha_dy + dphi_dir_tau0 * dtauD_dy;

        dphi_ris_dx = dphi_ris_beta0  * dbeta_dx  + dphi_ris_tau0 * dtauR_dx;
        dphi_ris_dy = dphi_ris_beta0  * dbeta_dy  + dphi_ris_tau0 * dtauR_dy;

        Jx = dphi_dir_dx*gdir + dphi_ris_dx*gris;
        Jy = dphi_dir_dy*gdir + dphi_ris_dy*gris;

        H = [real(Jx'*Jx), real(Jx'*Jy);
             real(Jy'*Jx), real(Jy'*Jy)];
        b = [real(Jx'*e);
             real(Jy'*e)];

        muEff = mu * max(1e-12, trace(H)/2);

        accepted = false;
        pos_old  = par.posT;

        for tr = 1:maxTrials
            delta = (H + muEff*eye(2)) \ b;
            pos_try = pos_old + delta;

            % Fast cost evaluation at pos_try
            [alpha_t, beta_t, tauD_t, tauR_t, ~, ~] = geom_from_pos(pos_try, posB, posR, d_BR_true, c);
            dAlpha_t = wrapToPi(alpha_t - alpha0);
            dBeta_t  = wrapToPi(beta_t  - beta0);
            dTauD_t  = (tauD_t - tauD0);
            dTauR_t  = (tauR_t - tauR0);

            phi_dir_t = phi_dir0 + dphi_dir_alpha0*dAlpha_t + dphi_dir_tau0*dTauD_t;
            phi_ris_t = phi_ris0 + dphi_ris_beta0 *dBeta_t  + dphi_ris_tau0*dTauR_t;

            Phi2_t = [phi_dir_t, phi_ris_t];
            g2_t   = (Phi2_t' * Phi2_t) \ (Phi2_t' * y_vec);
            e_t    = y_vec - Phi2_t*g2_t;
            Jtry   = real(e_t'*e_t);

            if Jtry < Jcur
                accepted = true;
                par.posT = pos_try;
                mu = max(1e-12, mu*muDn);
                break;
            else
                mu = mu * muUp;
                muEff = mu * max(1e-12, trace(H)/2);
            end
        end

        if ~accepted
            par.posT = pos_old;
        end

        % ===== Log =====
        [alpha_c, beta_c, tauD_c, tauR_c, ~, ~] = geom_from_pos(par.posT, posB, posR, d_BR_true, c);
        dAlpha = wrapToPi(alpha_c - alpha0);
        dBeta  = wrapToPi(beta_c  - beta0);
        dTauD  = (tauD_c - tauD0);
        dTauR  = (tauR_c - tauR0);

        phi_dir = phi_dir0 + dphi_dir_alpha0*dAlpha + dphi_dir_tau0*dTauD;
        phi_ris = phi_ris0 + dphi_ris_beta0 *dBeta  + dphi_ris_tau0*dTauR;

        Phi2 = [phi_dir, phi_ris];
        g2   = (Phi2' * Phi2) \ (Phi2' * y_vec);
        e    = y_vec - Phi2*g2;

        hist.J_total(step) = real(e'*e);
        hist.alpha(step)   = alpha_c;
        hist.beta(step)    = beta_c;
        hist.tau_dir(step) = tauD_c;
        hist.tau_ris(step) = tauR_c;
        hist.xT(step)      = par.posT(1);
        hist.yT(step)      = par.posT(2);
        hist.gdir_abs_pp(step)= abs(g2(1));
        hist.gris_abs_pp(step)= abs(g2(2));

        fprintf('Out%2d In%1d (step%3d) | J=%.3e | xT=%.4f yT=%.4f | alpha=%.2f° beta=%.2f° | tauD=%.4f tauR=%.4f | mu=%.2e\n', ...
            out, in, step, hist.J_total(step), par.posT(1), par.posT(2), ...
            rad2deg(alpha_c), rad2deg(beta_c), tauD_c, tauR_c, mu);
    end
end
toc

%% ================== Plots ==================
figure;
semilogy(hist.J_total,'-o','LineWidth',1.5); grid on;
title('Total cost','Interpreter','none');
xlabel('Inner step'); ylabel('Cost');

figure;
subplot(1,2,1);
plot(hist.xT,'-o'); hold on; yline(posT(1),'--'); grid on;
xlabel('Inner step'); ylabel('x'); title('Target x-coordinate');
legend('est','true');

subplot(1,2,2);
plot(hist.yT,'-o'); hold on; yline(posT(2),'--'); grid on;
xlabel('Inner step'); ylabel('y'); title('Target y-coordinate');
legend('est','true');

figure;
plot(hist.gdir_abs_pp,'-','LineWidth',1.5); hold on;
plot(hist.gris_abs_pp,'-','LineWidth',1.5); grid on;
legend('Direct path','RIS-assisted path');
xlabel('Iteration'); ylabel('|Gain|');
save('data_proposed.mat', '-struct', 'hist', 'gdir_abs_pp', 'gris_abs_pp');

%% ================== Local functions ==================
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

function [dalpha_dx, dalpha_dy, dtauD_dx, dtauD_dy, dbeta_dx,  dbeta_dy,  dtauR_dx, dtauR_dy] ...
    = geom_partials(posT, posB, posR, c)

    dB = posT - posB;  xB = dB(1); yB = dB(2);
    dR = posT - posR;  xR = dR(1); yR = dR(2);

    rB = norm(dB);
    rR = norm(dR);

    dalpha_dx = -yB/(rB^2);
    dalpha_dy =  xB/(rB^2);

    dbeta_dx  = -yR/(rR^2);
    dbeta_dy  =  xR/(rR^2);

    dtauD_dx  = (2/c) * (xB/rB);
    dtauD_dy  = (2/c) * (yB/rB);

    dtauR_dx  = (2/c) * (xR/rR);
    dtauR_dy  = (2/c) * (yR/rR);
end

function [phi, dphi_tau, dphi_alpha] = build_phi_dir_and_jac(alpha, tau, F, u_tx, u_rx, daULA, omega)
    [Nt, Ns, N] = size(F);
    Nr = Nt;

    phi        = zeros(Nr*Ns*N,1);
    dphi_tau   = zeros(size(phi));
    dphi_alpha = zeros(size(phi));

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
