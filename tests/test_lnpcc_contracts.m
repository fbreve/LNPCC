function tests = test_lnpcc_contracts
% Regression tests for graph reuse, label binding, RNG policy, and MEX.
% Run from the repository root: runtests('tests')
% MEX-dependent tests are skipped when lnpccloop is not compiled.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(root);
    testCase.TestData.root = root;
    [testCase.TestData.X, testCase.TestData.slabel] = wineFixture(root);
end

function testGraphFormatsAndSymmetry(testCase)
    X = testCase.TestData.X;
    slabel = testCase.TestData.slabel;
    optimized = lnpcc_buildgraph(X, slabel, k=10, disttype="seuclidean");
    globalScale = lnpcc_buildgraph(X, slabel, k=10, ...
        disttype="seuclidean", scale_scope="global");
    legacy = lnpcc_buildgraph(X, slabel, k=10, ...
        disttype="seuclidean", legacy_knn=true);
    assertGraph(testCase, optimized, slabel);
    assertGraph(testCase, globalScale, slabel);
    assertGraph(testCase, legacy, slabel);
    testCase.verifyEqual(optimized.scale_scope, "subset");
    testCase.verifyEqual(globalScale.scale_scope, "global");
    testCase.verifyEqual(legacy.legacy_knn, true);
end

function testDeterministicGraphConstruction(testCase)
    X = testCase.TestData.X;
    slabel = testCase.TestData.slabel;
    a = lnpcc_buildgraph(X, slabel, k=10, disttype="seuclidean");
    b = lnpcc_buildgraph(X, slabel, k=10, disttype="seuclidean");
    testCase.verifyEqual(a.nsize, b.nsize);
    testCase.verifyEqual(a.nlist, b.nlist);
end

function testRejectsMismatchedLabels(testCase)
    X = testCase.TestData.X;
    slabel = testCase.TestData.slabel;
    graph = lnpcc_buildgraph(X, slabel, k=10);
    altered = slabel;
    first = find(altered > 0, 1);
    altered(first) = mod(double(altered(first)), 3) + 1;
    testCase.verifyError(@() lnpcc(graph, altered, Xgraph=true), ...
        'LNPCC:GraphLabelsMismatch');
end

function testRejectsInvalidGraph(testCase)
    X = testCase.TestData.X;
    slabel = testCase.TestData.slabel;
    graph = lnpcc_buildgraph(X, slabel, k=10);
    graph.nlist(1, 1) = uint32(0);
    testCase.verifyError(@() lnpcc(graph, slabel, Xgraph=true), ...
        'LNPCC:InvalidGraph');
end

function testPrebuiltAndDirectGraphsAreNumericallyIdentical(testCase)
    assumeMex(testCase);
    X = testCase.TestData.X;
    slabel = testCase.TestData.slabel;
    G = lnpcc_buildgraph(X, slabel, k=10, disttype="seuclidean");
    [owner1, pot1, own1, dist1] = lnpcc(X, slabel, ...
        k=10, disttype="seuclidean", ...
        useseed=true, seed=1301, earlystop=false, maxiter=150);
    [owner2, pot2, own2, dist2] = lnpcc(G, slabel, ...
        Xgraph=true, useseed=true, seed=1301, ...
        earlystop=false, maxiter=150);
    testCase.verifyEqual(owner1, owner2);
    testCase.verifyEqual(pot1, pot2);
    testCase.verifyEqual(own1, own2);
    testCase.verifyEqual(dist1, dist2);
end

function testLegacyPrebuiltAndDirectAreNumericallyIdentical(testCase)
    assumeMex(testCase);
    X = testCase.TestData.X;
    slabel = testCase.TestData.slabel;
    G = lnpcc_buildgraph(X, slabel, k=10, ...
        disttype="seuclidean", legacy_knn=true);
    [owner1, pot1, own1, dist1] = lnpcc(X, slabel, ...
        k=10, disttype="seuclidean", legacy_knn=true, ...
        useseed=true, seed=1302, earlystop=false, maxiter=100);
    [owner2, pot2, own2, dist2] = lnpcc(G, slabel, ...
        Xgraph=true, useseed=true, seed=1302, ...
        earlystop=false, maxiter=100);
    testCase.verifyEqual(owner1, owner2);
    testCase.verifyEqual(pot1, pot2);
    testCase.verifyEqual(own1, own2);
    testCase.verifyEqual(dist1, dist2);
end

function testSeedOnceIsDeterministicAndDiffersFromLegacyReseeding(testCase)
    assumeMex(testCase);
    X = testCase.TestData.X;
    slabel = testCase.TestData.slabel;
    [o1, p1, d1, r1] = lnpcc(X, slabel, k=10, ...
        useseed=true, seed=1303, earlystop=false, maxiter=200);
    [o2, p2, d2, r2] = lnpcc(X, slabel, k=10, ...
        useseed=true, seed=1303, earlystop=false, maxiter=200);
    testCase.verifyEqual(o1, o2);
    testCase.verifyEqual(p1, p2);
    testCase.verifyEqual(d1, d2);
    testCase.verifyEqual(r1, r2);
    [~, ~, legacy1] = lnpcc(X, slabel, k=10, ...
        useseed=true, seed=1303, reseed_each_run=true, ...
        earlystop=false, maxiter=200);
    [~, ~, legacy2] = lnpcc(X, slabel, k=10, ...
        useseed=true, seed=1303, reseed_each_run=true, ...
        earlystop=false, maxiter=200);
    testCase.verifyEqual(legacy1, legacy2);
    testCase.verifyNotEqual(d1, legacy1);
end

function assumeMex(testCase)
    testCase.assumeTrue(exist('lnpccloop', 'file') == 3, ...
        'Compile the platform-specific MEX: mex lnpccloop.c');
end

function assertGraph(testCase, G, slabel)
    n = numel(slabel);
    testCase.verifyClass(G.nlist, 'uint32');
    testCase.verifyClass(G.nsize, 'uint16');
    testCase.verifyEqual(G.slabel(:), slabel(:));
    testCase.verifyEqual(size(G.nlist, 1), n);
    A = graphAdjacency(G);
    testCase.verifyTrue(isequal(A, A'), 'Graph must be symmetric.');
    testCase.verifyEqual(nnz(diag(A)), 0, 'No self-loops allowed.');
    testCase.verifyEqual(full(sum(A, 2)), double(G.nsize(:)));
end

function A = graphAdjacency(G)
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

function [X, slabel] = wineFixture(root)
    wine = load(fullfile(root, 'wine.data'), '-ascii');
    X = wine(:, 2:end);
    classes = uint16(wine(:,1));
    slabel = zeros(size(classes), 'uint16');
    for c = 1:3
        inds = find(classes == c, 3, 'first');
        slabel(inds(1:2)) = c;
    end
    % Intentionally include one incorrect label to exercise LNPCC graphs.
    inds = find(classes == 1, 3, 'first');
    slabel(inds(3)) = 2;
end
