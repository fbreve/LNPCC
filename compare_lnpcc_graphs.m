function report = compare_lnpcc_graphs()
% Compare label-aware graph construction modes without running PCC.
% MATLAB R2021a+, Statistics and Machine Learning Toolbox required.
% Usage:
%   report = compare_lnpcc_graphs();
%
% This diagnostic reports graph differences. It does NOT equate a different
% graph with an incorrect one, and does not claim equal classification.
    root = fileparts(mfilename('fullpath'));
    wine = load(fullfile(root, 'wine.data'), '-ascii');
    X = wine(:, 2:end);
    label = uint16(wine(:,1));
    slabel = zeros(size(label), 'uint16');
    for c = 1:3
        indices = find(label == c, 13, 'first');
        slabel(indices(1:12)) = c;
    end
    % Keep a controlled incorrect label so the label-aware logic is used.
    indices = find(label == 1, 3, 'first');
    slabel(indices(3)) = 2;
    k = 10;

    tic;
    historical = lnpcc_buildgraph(X, slabel, k=k, ...
        disttype="seuclidean", scale_scope="subset");
    tSubset = toc;
    tic;
    correctedScale = lnpcc_buildgraph(X, slabel, k=k, ...
        disttype="seuclidean", scale_scope="global");
    tGlobal = toc;
    tic;
    legacy = lnpcc_buildgraph(X, slabel, k=k, ...
        disttype="seuclidean", legacy_knn=true);
    tLegacy = toc;

    A = toAdjacency(historical);
    B = toAdjacency(correctedScale);
    C = toAdjacency(legacy);
    if ~isequal(A, A') || ~isequal(B, B') || ~isequal(C, C')
        error('LNPCC:AsymmetricGraph', ...
            'A graph builder produced a non-reciprocal adjacency list.');
    end
    report = table( ...
        ["optimized_subset"; "optimized_global"; "legacy"], ...
        [nnz(triu(A,1)); nnz(triu(B,1)); nnz(triu(C,1))], ...
        [tSubset; tGlobal; tLegacy], ...
        [nnz(triu(xor(A,C),1)); nnz(triu(xor(B,C),1)); 0], ...
        'VariableNames', {'Method','Edges','Seconds','DifferentEdgesVsLegacy'});
    disp(report);
    fprintf('Different edges (subset vs global): %d\n', ...
        nnz(triu(xor(A,B),1)));
    fprintf('These numbers characterize graphs, not classification accuracy.\n');
end

function A = toAdjacency(G)
    n = size(G.nlist,1);
    rows = [];
    cols = [];
    for i = 1:n
        neighbors = double(G.nlist(i, 1:double(G.nsize(i))));
        rows = [rows; repmat(i, numel(neighbors), 1)]; %#ok<AGROW>
        cols = [cols; neighbors(:)]; %#ok<AGROW>
    end
    A = spones(sparse(rows, cols, 1, n, n));
end
