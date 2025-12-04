close all
data_dir = "./data/";


figure()
hold on

ms = 12;
lw = 5;

n_c = 5;
n_bot = 2;
n_top = 2;
n_tot = n_c + n_bot + n_top;
colors = ice(n_tot); % from cmap: https://github.com/tsipkens/cmap
colors = colors(n_bot:end-n_top, :);
colors = flipud(colors);
beta_bright = 0.3;
beta_ref = -0.4;

ref_colors = colors;

val = 0.1;
black = [val, val, val];


n_runs = 3;
[phi, dat, var_dat] = average_data(data_dir, n_runs);

std_err = 2*var_dat / sqrt(n_runs);

handles = [];
for i=1:5
  % h = errorbar(phi, 1+dat(:,i), std_err(:,i), Color=colors(i, :), Marker='square', MarkerFaceColor=colors(i, :));
  h = plot(phi, dat(:,i), Color=brighten(colors(i, :), beta_bright), Marker='square', MarkerFaceColor=brighten(colors(i, :), ...
    beta_bright), MarkerSize=ms, LineWidth=lw);
  handles = [handles, h];
end

xticks(round(phi,2))
xtickformat("%.2f")
labels = ["0.05" "0.10" " " "0.20" " " "0.30" " " "0.40" " " "0.50"];
xticklabels(labels)
xtickangle(-45)

box on
daspect([0.5, 10, 1])
grid on
xscale("log")
yscale("log")

fs=30;
xlabel("Packing fraction, $\phi$", fontsize=fs)
ylabel("Runtime (s)", fontsize=fs)
legend(handles, ["m=12", "m=42", "m=162", "m=642", "m=2562"], Location="northwest")
ax = gca;
ax.FontSize = 30;
xlim([0, 0.5])

function [phi, avg_dat, var_dat]  = average_data(data_dir, n_avg)


dat = readmatrix(data_dir + "runtime_1.csv");
all_dat = zeros([3, size(dat)] - [0,0,1]);

for i=1:n_avg
  fname = data_dir + "runtime_" + num2str(i) + ".csv";
  dat = readmatrix(fname);
  phi = dat(:,1);
  dat = dat(:,2:end);
  all_dat(i, :, :) = dat;
end

avg_dat = mean(all_dat, 1);
var_dat = var(all_dat, 0, 1);

avg_dat = squeeze(avg_dat);
var_dat = squeeze(var_dat);
end
