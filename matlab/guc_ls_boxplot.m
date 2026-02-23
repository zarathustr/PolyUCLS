%% guc_ls_boxplot.m
% Boxplot drawing script for the C++ benchmark "benchmark_noise_sweep".
%
% Expected CSV columns (produced by benchmark_noise_sweep):
%   problem, solver, noise, trial, rot_err_deg, trans_err, orth_err
%
% Typical usage:
%   1) Build + run the benchmark from C++ (in project root):
%        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
%        cmake --build build -j
%        ./build/benchmark_noise_sweep --mc 200 --noise 0,0.002,0.005,0.01,0.02 --out build/guc_ls_noise_sweep.csv
%   2) Run this script in MATLAB.

clear; close all; clc;
warning('off');
format long g

%% User settings
csvFile = '../build/guc_ls_noise_sweep.csv';
saveFigures = false;
outDir = '.';

%% Read CSV
if ~isfile(csvFile)
    error('CSV file not found: %s', csvFile);
end
T = readtable(csvFile);

T.problem = string(T.problem);
T.solver  = string(T.solver);

problems = unique(T.problem, 'stable');

%% Plot style
FontSizeAxis   = 18;
FontSizeLegend = 16;
BoxLineWidth   = 2.0;
LegendLoc      = 'northwest';
boxWidth       = 0.18;

cGray   = [0.5, 0.5, 0.5];
cBlue   = [0.0, 0.4470, 0.7410];
cOrange = [0.8500, 0.3250, 0.0980];
cPurple = [0.4940, 0.1840, 0.5560];
cGreen  = [0.4660, 0.6740, 0.1880];
cRed    = [1.0, 0.0, 0.0];

%% Loop problems
for p = 1:numel(problems)
    prob = problems(p);
    Tp = T(T.problem == prob, :);
    if height(Tp) == 0
        continue;
    end

    %% Decide default method order + labels based on problem name
    if startsWith(prob, "unitnorm")
        n = str2double(extractAfter(prob, "unitnorm"));
        methods = ["NormLS", "KKTNewton", "PolyRoot"];
        DeviceColors = {cGray, cBlue, cRed};
        probTitle = sprintf('$\\|u\\|=1$ (vector, $n=%d$)', n);
        y1 = 'Angle Error (deg)';
        y2 = 'Residual $\\|Bu-g\\|^2$';
        y3 = 'Orthogonality $|u^\\top u-1|$';

    elseif startsWith(prob, "procrustes")
        n = str2double(extractAfter(prob, "procrustes"));
        if n == 4
            % Small case: include the multi-solution representative and the Newton baseline.
            methods = ["LSProject", "PolarSqrt", "GBEnum", "ProcrustesSVD", "StiefelNewton"];
            DeviceColors = {cGray, cBlue, cPurple, cRed, cGreen};
        else
            % Large cases: keep only scalable representatives.
            methods = ["LSProject", "PolarSqrt", "ProcrustesSVD"];
            DeviceColors = {cGray, cBlue, cRed};
        end
        probTitle = sprintf('$X^\\top X = I$ (matrix, $%d\\times %d$)', n, n);
        y1 = 'Geodesic Error (deg)';
        y2 = 'Residual $\\|AX-B\\|_F^2$';
        y3 = 'Orthogonality $\\|X^\\top X-I\\|_F$';

    else
        methods = unique(Tp.solver, 'stable');
        baseCols = {cGray, cOrange, cBlue, cPurple, cGreen, cRed};
        DeviceColors = baseCols(1:min(numel(baseCols), numel(methods)));
        probTitle = char(prob);
        y1 = 'rot\_err\_deg';
        y2 = 'trans\_err';
        y3 = 'orth\_err';
    end

    % Keep only methods that exist in the CSV for this problem
    hasMethod = ismember(methods, unique(Tp.solver));
    methods = methods(hasMethod);
    DeviceColors = DeviceColors(hasMethod);

    noiseLevels = unique(double(Tp.noise));
    noiseLevels = sort(noiseLevels);
    xBase = 1:numel(noiseLevels);
    xTickLabels = arrayfun(@(x) sprintf('%.4g', x), noiseLevels, 'UniformOutput', false);

    M = numel(methods);
    offsets = linspace(-0.28, 0.28, max(1,M));
    offsets = offsets(1:M);

    %% Figure 1: rot_err_deg
    figure('Color','w'); hold on; set(gca,'NextPlot','add');
    plotGroupedBoxplot(Tp, methods, DeviceColors, noiseLevels, xBase, offsets, ...
        "rot_err_deg", boxWidth, BoxLineWidth);
    addLegendLines(methods, DeviceColors);
    xlabel('Noise Levels', 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
    ylabel(y1, 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
    title(probTitle, 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
    grid on; grid minor;
    set(gca, 'XTick', xBase, 'XTickLabel', xTickLabels, 'FontSize', FontSizeAxis);
    xlim([0.5, numel(noiseLevels)+0.5]);
    legend(methods, 'Interpreter','none', 'FontSize', FontSizeLegend, 'Location', LegendLoc);
    if saveFigures
        if ~exist(outDir,'dir'), mkdir(outDir); end
        saveas(gcf, fullfile(outDir, sprintf('%s_rot_boxplot.png', char(prob))));
    end

    %% Figure 2: trans_err (residual)
    figure('Color','w'); hold on; set(gca,'NextPlot','add');
    plotGroupedBoxplot(Tp, methods, DeviceColors, noiseLevels, xBase, offsets, ...
        "trans_err", boxWidth, BoxLineWidth);
    addLegendLines(methods, DeviceColors);
    xlabel('Noise Levels', 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
    ylabel(y2, 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
    title(probTitle, 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
    grid on; grid minor;
    set(gca, 'XTick', xBase, 'XTickLabel', xTickLabels, 'FontSize', FontSizeAxis);
    xlim([0.5, numel(noiseLevels)+0.5]);
    legend(methods, 'Interpreter','none', 'FontSize', FontSizeLegend, 'Location', LegendLoc);
    if saveFigures
        if ~exist(outDir,'dir'), mkdir(outDir); end
        saveas(gcf, fullfile(outDir, sprintf('%s_cost_boxplot.png', char(prob))));
    end

    %% Figure 3: orth_err (constraint satisfaction)
    if any(string(Tp.Properties.VariableNames) == "orth_err")
        figure('Color','w'); hold on; set(gca,'NextPlot','add');
        plotGroupedBoxplot(Tp, methods, DeviceColors, noiseLevels, xBase, offsets, ...
            "orth_err", boxWidth, BoxLineWidth);
        addLegendLines(methods, DeviceColors);
        xlabel('Noise Levels', 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
        ylabel(y3, 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
        title(probTitle, 'Interpreter','LaTeX', 'FontSize', FontSizeAxis);
        grid on; grid minor;
        set(gca, 'XTick', xBase, 'XTickLabel', xTickLabels, 'FontSize', FontSizeAxis);
        xlim([0.5, numel(noiseLevels)+0.5]);
        legend(methods, 'Interpreter','none', 'FontSize', FontSizeLegend, 'Location', LegendLoc);
        if saveFigures
            if ~exist(outDir,'dir'), mkdir(outDir); end
            saveas(gcf, fullfile(outDir, sprintf('%s_orth_boxplot.png', char(prob))));
        end
    end
end

%% -------- Local helpers (style adapted from various_c3p_boxplot.m) --------

function plotGroupedBoxplot(Tp, methods, DeviceColors, noiseLevels, xBase, offsets, metricName, boxWidth, lineW)
M = numel(methods);
K = numel(noiseLevels);

for m = 1:M
    method = methods(m);
    color = DeviceColors{m};

    maxCount = 0;
    for k = 1:K
        idx = string(Tp.solver) == method & abs(double(Tp.noise) - noiseLevels(k)) < 1e-12;
        vals = Tp.(metricName)(idx);
        maxCount = max(maxCount, numel(vals));
    end

    Xmat = NaN(maxCount, K);
    for k = 1:K
        idx = string(Tp.solver) == method & abs(double(Tp.noise) - noiseLevels(k)) < 1e-12;
        vals = Tp.(metricName)(idx);
        if ~isempty(vals)
            Xmat(1:numel(vals), k) = vals(:);
        end
    end

    pos = xBase + offsets(m);

    h = boxplot(Xmat, 'positions', pos, 'colors', color, ...
        'symbol', '', 'widths', boxWidth, 'plotstyle', 'traditional');

    try
        set(h, 'LineWidth', lineW);
    catch
    end
    try
        set(h(1:2,:), 'LineStyle', '--');
    catch
    end
    try
        set(h(3:4,:), 'LineStyle', '--');
    catch
    end
end
end

function addLegendLines(methods, DeviceColors)
for i = 1:numel(methods)
    plot(NaN, 1, 'color', DeviceColors{i}, 'LineWidth', 1);
end
end
