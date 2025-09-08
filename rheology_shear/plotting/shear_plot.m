ref_dir = "extracted_data/";
data_dir = "data/";

figure()
hold on

% colors = ["#9a090c",
% "#2d16bd",
% "#7b00b8",
% "#e93172",
% "#4b94de",
% "#d96ff2",
% "#ec8633",
% "#55c743"];
ms = 12;
lw = 5;

n_c = 5;
n_bot = 2;
n_top = 2;
n_tot = n_c + n_bot + n_top;
colors = ice(n_tot);
colors = colors(n_bot:end-n_top, :);
colors = flipud(colors);
beta_bright = 0.3;
beta_ref = -0.4;

% ref_colors = oslo(6);
% ref_colors = ref_colors(3:end-1, :);
% ref_colors = matter(n_tot-2);
% ref_colors = ref_colors(n_bot:end-n_top, :);
% ref_colors = flipud(ref_colors);

ref_colors = colors;

val = 0.1;
black = [val, val, val];


% dat = readmatrix(data_dir + "shear_stress_1.csv");
n_runs = 3;
[phi, dat, var_dat] = average_data(data_dir, n_runs);

std_err = 2*var_dat / sqrt(n_runs);

handles = [];
for i=1:5
  % h = errorbar(phi, 1+dat(:,i), std_err(:,i), Color=colors(i+3), Marker='square', MarkerFaceColor=colors(i+3));
  h = plot(phi, 1+dat(:,i), Color=brighten(colors(i, :), beta_bright), Marker='square', MarkerFaceColor=brighten(colors(i, :), ...
    beta_bright), MarkerSize=ms, LineWidth=lw);
  handles = [handles, h];
end

i=1;
ref_handles = [];
for n_ref = [12,42,162]
  filename = ref_dir + num2str(n_ref) + "-bead.csv";
  data = readmatrix(filename);
  phi = data(:,1);
  eta = data(:,2);
  h = plot(phi, eta, Color=brighten(ref_colors(i, :), beta_ref), Marker="diamond", MarkerFaceColor=brighten(ref_colors(i, :), beta_ref), ...
    LineStyle="--", MarkerSize=ms, MarkerEdgeColor="none", LineWidth=lw-2);
  ref_handles = [ref_handles, h];
  i = i +1;
end

phi_grid = linspace(0, 0.5, 200);
eta_func = @(p) (1+1.5*p.*(1+p.*(1+p-2.3*p.^2))) ./ (1 - p.*(1+p.*(1+p-2.3*p.^2)));
eta_analytic = eta_func(phi_grid);

h_ladd = plot(phi_grid, eta_analytic, '-k');

% temp = readmatrix("temp.csv");
% plot(temp(:,1), 1+temp(:,2), '--k');

ylim([0.5, 7.5]);
xlim([0, 0.5])

box on
daspect([0.5, 7, 1])
grid on

xlabel("$\phi$")
ylabel("$\eta_r$")
% handles = [h_ladd, ref_handles, handles];
handles = [handles, ref_handles, h_ladd];
% legend(handles, ["Ladd (1990)", "m=12 (ref.)", "m=42 (ref.)", "m=162 (ref.)", "m=12", "m=42", "m=162", "m=642", "m=2562"], Location="northwest", NumColumns=2)
legend(handles, ["m=12", "m=42", "m=162", "m=642", "m=2562", "m=12 (ref.)", "m=42 (ref.)", "m=162 (ref.)",  "Ladd (1990)"], Location="northwest", NumColumns=2)

xticks(0:0.1:0.5)
yticks(1:7)

function [phi, avg_dat, var_dat]  = average_data(data_dir, n_avg)


dat = readmatrix(data_dir + "shear_stress_1.csv");
all_dat = zeros([3, size(dat)] - [0,0,1]);

for i=1:n_avg
fname = data_dir + "shear_stress_" + num2str(i) + ".csv";
dat = readmatrix(fname)
phi = dat(:,1);
dat = dat(:,2:end);
all_dat(i, :, :) = dat;
end

avg_dat = mean(all_dat, 1);
var_dat = var(all_dat, 0, 1);

avg_dat = squeeze(avg_dat);
var_dat = squeeze(var_dat);
end
