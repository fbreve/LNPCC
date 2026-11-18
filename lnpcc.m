% Particle Competition and Cooperation for Semi-Supervised Learning with
% Label Noise
% by Fabricio Breve - 24/01/2019
%
% If you use this algorithm, please cite:
% BREVE, Fabricio Aparecido; ZHAO, Liang; QUILES, Marcos Gonçalves. 
% Particle Competition and Cooperation for Semi-Supervised Learning with
% Label Noise. Neurocomputing (Amsterdam), v.160, p.63 – 72, 2015.
%
% Usage: [owner, pot, owndeg, distnode] = lnpcc(X, slabel, options)
% INPUT:
% X         - Matrix where each line is a data item and each column is an
%             attribute
% slabel    - Vector where each element is the label of the corresponding
%             data item in X (use 1,2,3,... for labeled data items and 0
%             for unlabeled data items)
% OPTIONS:
% k         - Each node is connected to its k-nearest neighbors. Default:
%             size of the dataset multiplied by 0.05.
% disttype  - Use 'euclidean', 'seuclidean', etc. Default: 'euclidean'.
%             See the MATLAB knnsearch function help for all the options.
% earlystop - Default: true.  Early stop is adjusted by 'valpha'. Set it 
%             to false to run exactly 'maxiter' iterations.
% valpha    - Lower it to stop earlier, accuracy may be lower. Default:
%             2000.
% pgrd      - Check p_grd in [1]. Default: 0.5.
% deltav    - Check delta_v in [1]. Default: 0.1
% deltap    - Default: 1 (leave it on default to match equations in [1])
% dexp      - Default: 2 (leave it on default to match equations in [1])
% nclass    - Amount of classes on the problem. Default is the highest
%             label number in slabel.
% maxiter   - Maximum amount of iterations. Default is 500,000.
% Execution - Always uses the compiled lnpccloop MEX routine.
%             There is no pure-MATLAB (mex=false) option in this version.
% useseed   - When true, calls C srand(seed) in the MEX routine.
%             Default: false. The seed is reset on each of the 10 runs.
% seed      - Seed for the C rand()/srand() generator (default: 0).
%             Fixed seeds can repeat the same trajectory across runs.
% Xgraph    - Reserved but currently unused. This version always
%             constructs a graph from a feature matrix X.
% legacy_knn- Default: false. When true, uses the original graph
%             construction based on the full distance matrix. The newer
%             knnsearch implementation may produce different neighbors.
%
% OUTPUT:
% owner     - vector of classes assigned to each data item
% pot       - dominance matrix from the LAST of the 10 runs. Owner is
%             determined from the sum of dominance matrices across runs.
% owndeg    - normalized fuzzy ownership accumulated across all 10 runs,
%             one row per sample and one column per class.
% distnode  - particle distance matrix from the LAST run.
%
% [1] BREVE, Fabricio Aparecido; Zhao, Liang; QUILES, Marcos Gonçalves;
% PEDRYCZ, Witold; LIU, Jiming, "Particle Competition and Cooperation in
% Networks for Semi-Supervised Learning," Knowledge and Data Engineering,
% IEEE Transactions on , vol.24, no.9, pp.1686,1698, Sept. 2012.
% doi: 10.1109/TKDE.2011.119
%
% [2] BREVE, Fabricio Aparecido; ZHAO, Liang. 
% "Fuzzy community structure detection by particle competition and cooperation."
% Soft Computing (Berlin. Print). , v.17, p.659 - 673, 2013.
%
% [3] BREVE, Fabricio Aparecido; ZHAO, Liang; QUILES, Marcos Gonçalves. 
% Particle Competition and Cooperation for Semi-Supervised Learning with
% Label Noise. Neurocomputing (Amsterdam), v.160, p.63 – 72, 2015.


function [owner, pot, owndeg, distnode] = lnpcc(X, slabel, options)
    arguments
        X 
        slabel uint16
        options.k uint16 = size(X,1)*0.05
        options.disttype string = 'euclidean'
        options.earlystop logical = true
        options.valpha double = 2000
        options.pgrd double = 0.500 % probability of taking the greedy movement
        options.deltav double = 0.100 % controls node domination levels increase/decrease rate
        options.deltap double = 1.000 % controls particle potential increase/decrease rate
        options.dexp double = 2 % probabilities exponential
        options.nclass uint16 = max(slabel) % quantity of classes
        options.maxiter uint32 = 500000 % maximum amount of iterations
        options.useseed logical = false % do not set seed
        options.seed int32 = 0 % random seed
        options.reseed_each_run logical = false % compatibility mode: repeat C RNG stream
        options.Xgraph logical = false % X is a feature matrix by default
        options.legacy_knn logical = false % legacy function to find k-nearest neighbors
    end   

    % A graph produced by lnpcc_buildgraph is tied to its input labels.
    if options.Xgraph
        G = X;
        validate_lnpcc_graph(G, slabel);
        qtnode = size(G.nlist, 1);
    else
        G = lnpcc_buildgraph(X, slabel, k=options.k, ...
            disttype=options.disttype, legacy_knn=options.legacy_knn);
        qtnode = size(X, 1);
    end

    % Keep the original padded, dense adjacency representation used by MEX.
    nsize = G.nsize;
    nlist = G.nlist;
    k = G.k;

    % tratamento da entrada
    slabel = uint16(slabel);
    % constantes
    potmax = 1.000; % potencial máximo
    potmin = 0.000; % potencial mínimo
    npart = sum(slabel~=0);
    if npart == 0 || options.nclass < 2
        error('LNPCC:InvalidClasses', ...
            'LNPCC requires labeled vertices from at least two classes.');
    end
    %stopmax = round((qtnode/npart)*round(options.valpha*0.01)); % qtde de iterações para verificar convergência
    stopmax = round((qtnode/double(npart*k))*round(options.valpha*0.1));

    % definindo classe de cada partícula
    partclass = slabel(slabel~=0);
    % definindo nó casa da partícula
    partnode = uint32(find(slabel));


    % definindo grau de propriedade
    potacc = zeros(qtnode,options.nclass);  % não podemos usar 0, porque nós não visitados dariam divisão por 0
    % inicializando tabela de potenciais com tudo igual
    potini = repmat(potmax/double(options.nclass),qtnode,options.nclass);
    % zerando potenciais dos nós rotulados
    potini(partnode,:) = 0;
    % ajustando potencial da classe respectiva do nó rotulado para 1      
    potini(sub2ind(size(potini),partnode,slabel(partnode))) = 1;

    % initializing the accumulated dominance vectors
    % we can't use 0, otherwise non-visited nodes would trigger a division
    % by zero.
    owndeg = repmat(realmin,qtnode,options.nclass);  

    for ri=1:10
        % ajustando potenciais para configuração inicial
        pot = potini;
        % definindo potencial da partícula em 1
        potpart = ones(1,npart);    
        % ajustando todas as distâncias na máxima possível
        distnode = repmat(min(intmax('uint8'),uint8(qtnode-1)),qtnode,npart);        
        % ajustando para zero a distância de cada partícula para seu
        % respectivo nó casa
        distnode(sub2ind(size(distnode),partnode',1:npart)) = 0;
        % colocando cada nó em sua casa
        partpos = partnode;

        % Deterministic mode seeds once and then advances the C RNG across
        % the ten runs. reseed_each_run=true restores the old repeated
        % stream for reproducing results from earlier code revisions.
        seed_this_run = options.useseed && ...
            (ri == 1 || options.reseed_each_run);
        lnpccloop(options.maxiter, npart, options.nclass, options.earlystop, ...
            stopmax, options.pgrd, options.dexp, options.deltav, ...
            options.deltap, potmin, partpos, partclass, potpart, ...
            slabel, nsize, distnode, nlist, pot, owndeg, ...
            seed_this_run, options.seed);

        potacc = potacc + pot;
    end
    [~,owner] = max(potacc,[],2);
    owndeg = owndeg ./ repmat(sum(owndeg,2),1,options.nclass);
end


% Verify the graph contract before passing pointer-sized data to MEX.
% In particular, the original input labels must match exactly: graphs
% cannot be reused across different noisy-label splits.
function validate_lnpcc_graph(G, slabel)
    fields = {'nsize', 'nlist', 'k', 'slabel'};
    if ~isstruct(G) || ~isscalar(G) || ~all(isfield(G, fields))
        error('LNPCC:InvalidGraph', ...
            'Xgraph=true requires a graph built with lnpcc_buildgraph.');
    end
    n = numel(slabel);
    if ~isa(G.nsize, 'uint16') || ~isvector(G.nsize) || ...
            numel(G.nsize) ~= n || ...
            ~isa(G.nlist, 'uint32') || size(G.nlist,1) ~= n || ...
            ~isa(G.k, 'uint16') || ~isscalar(G.k) || ...
            double(G.k) < 1 || double(G.k) >= n
        error('LNPCC:InvalidGraph', 'Invalid graph dimensions or data types.');
    end
    if ~isequal(uint16(G.slabel(:)), slabel(:))
        error('LNPCC:GraphLabelsMismatch', ...
            'Graph was constructed with different labels; rebuild it.');
    end
    if max(double(G.nsize)) > size(G.nlist,2) || any(G.nsize == 0)
        error('LNPCC:InvalidGraph', 'Graph has invalid node degrees.');
    end
    for v = 1:n
        neighbors = double(G.nlist(v,1:double(G.nsize(v))));
        if any(neighbors < 1 | neighbors > n | neighbors == v) || ...
                numel(unique(neighbors)) ~= numel(neighbors)
            error('LNPCC:InvalidGraph', ...
                'Graph has an invalid or repeated neighbor at node %d.', v);
        end
    end
end
