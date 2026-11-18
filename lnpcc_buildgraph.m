% LNPCC_BUILDGRAPH Construct the label-aware nearest-neighbor network.
%
% G = lnpcc_buildgraph(X, slabel, k=10, disttype="seuclidean")
% G = lnpcc_buildgraph(X, slabel, k=10, legacy_knn=true)
%
% IMPORTANT: The network is built using the provided labels (including
% incorrect/noisy labels). Reuse it only with the SAME slabel vector.
% Data structure deliberately uses the original padded dense nlist matrix.
%
function G = lnpcc_buildgraph(X, slabel, options)
    arguments
        X
        slabel uint16
        options.k uint16 = size(X,1)*0.05
        options.disttype string = 'euclidean'
        options.legacy_knn logical = false
        options.scale_scope string = 'global'
    end

    if ~isnumeric(X) || ~ismatrix(X) || isempty(X)
        error('LNPCC:InvalidFeatures', ...
            'X must be a nonempty numeric feature matrix.');
    end
    qtnode = size(X,1);
    if ~isvector(slabel) || numel(slabel) ~= qtnode
        error('LNPCC:InvalidLabels', ...
            'slabel must contain one class label per row of X.');
    end
    if options.k < 1 || double(options.k) >= qtnode
        error('LNPCC:InvalidK', ...
            'k must satisfy 1 <= k < number of samples.');
    end
    slabel = slabel(:);
    k = uint16(options.k);

    if ~ismember(options.scale_scope, ["subset", "global"])
        error('LNPCC:InvalidScaleScope', ...
            'scale_scope must be "subset" or "global".');
    end

    % The historical optimized builder computes seuclidean scales from
    % X_same and X_other independently. The legacy builder uses pdist(X),
    % which uses GLOBAL feature scales. For controlled comparisons,
    % scale_scope="global" normalizes X once and uses Euclidean kNN.
    % Use "subset" explicitly to restore the 2025 optimized behavior.
    Xsearch = X;
    search_disttype = options.disttype;
    if ~options.legacy_knn && options.scale_scope == "global" && ...
            options.disttype == "seuclidean"
        scale = std(double(X), 0, 1);
        if any(~isfinite(scale) | scale <= 0)
            error('LNPCC:InvalidScale', ...
                'Global standardized Euclidean needs positive finite feature scales.');
        end
        Xsearch = double(X) ./ scale;
        search_disttype = 'euclidean';
    end

    if options.legacy_knn
        [nsize, nlist] = legacy_build_knn_with_labels( ...
            X, slabel, k, options.disttype);
    else
        [nsize, nlist] = build_knn_with_labels( ...
            Xsearch, slabel, k, search_disttype);
    end

    G.nsize = nsize;
    G.nlist = nlist;
    G.k = k;
    G.slabel = slabel;
    G.disttype = options.disttype;
    G.legacy_knn = options.legacy_knn;
    G.scale_scope = options.scale_scope;
end

function [nsize, nlist] = build_knn_with_labels(X, slabel, k, disttype)
%
% Generated with Perplexity, based on the current (2025) implementation of 
% finding k-nearest neighbors I wrote for the original PCC.
% It does not exactly match the neighbors list produced by the legacy
% method, probably due to how knnsearch() handles ties.
%
% In larger datasets, this method is much faster than calculating all
% the distances like the original version did.
% 
% Constrói listas de vizinhos com:
% - Prioridade forte para vizinhos de mesmo rótulo (se houver >= k)
% - Reciprocidade: se j é vizinho de i, então i entra como vizinho de j.
%
% X      : qtnode x d
% slabel: qtnode x 1, 0 = não rotulado, >0 = rótulo
% k      : número alvo de vizinhos
% disttype: tipo de distância aceito por knnsearch

    qtnode = size(X,1);
    k      = double(k);
    slabel = double(slabel(:));

    % ---------------------------
    % 1) Listas iniciais (sem reciprocidade)
    % ---------------------------
    nsize = uint16(zeros(qtnode,1));
    nlist = zeros(qtnode, k, 'uint32');

    % 1a) nós não rotulados: k vizinhos mais próximos em todo X
    idx_unl = find(slabel == 0);
    if ~isempty(idx_unl)
        K_all = knnsearch(X, X(idx_unl,:), ...
                          'K', k+1, ...
                          'Distance', disttype);
        % remove self (assumindo que sai na 1ª coluna)
        K_all = K_all(:,2:k+1);
        nsize(idx_unl) = uint16(k);
        nlist(idx_unl,:) = uint32(K_all);
    end

    % 1b) nós rotulados: priorizar vizinhos do mesmo rótulo
    labels = unique(slabel(slabel>0))';
    for c = labels
        idx_c = find(slabel == c);
        if isempty(idx_c), continue; end

        same_mask  = (slabel == c);
        same_inds  = find(same_mask);      % nós com mesmo rótulo
        other_inds = find(~same_mask);     % demais nós

        X_same  = X(same_inds,:);
        X_other = X(other_inds,:);

        % kNN só dentro do mesmo rótulo (todas queries de uma vez)
        k_same_max = min(k+1, size(X_same,1));
        Idx_same_local = knnsearch(X_same, X(idx_c,:), ...
                                   'K', k_same_max, ...
                                   'Distance', disttype);
        Idx_same_global = same_inds(Idx_same_local);

        for t = 1:numel(idx_c)
            this_node = idx_c(t);
            row = Idx_same_global(t,:);

            % remove self se aparecer
            row(row == this_node) = [];
            row = unique(row, 'stable');

            if numel(row) >= k
                row   = row(:).';
                neigh = row(1:k);
            else
                need  = k - numel(row);
                extra = [];

                if ~isempty(X_other) && need > 0
                    k_other = min(need, size(X_other,1));
                    Idx_other_local = knnsearch(X_other, X(this_node,:), ...
                                                'K', k_other, ...
                                                'Distance', disttype);
                    extra = other_inds(Idx_other_local);
                end

                row   = row(:).';
                extra = extra(:).';
                neigh = [row, extra];
            end

            nsize(this_node) = uint16(numel(neigh));
            nlist(this_node,1:numel(neigh)) = uint32(neigh);
        end
    end

    % ---------------------------
    % 2) Garantir reciprocidade
    % ---------------------------

    % reserva espaço extra para conexões recíprocas
    nlist(:,end+1:end+k) = uint32(0);
    nsize_full = double(nsize);   % trabalhar em double para indexar

    for i = 1:qtnode
        deg_i  = nsize_full(i);
        if deg_i == 0, continue; end
        neigh_i = nlist(i,1:deg_i);
        for v = 1:numel(neigh_i)
            j = neigh_i(v);
            if j == 0, continue; end
            pos = nsize_full(j) + 1;
            % aumenta largura se necessário
            if pos > size(nlist,2)
                nlist(:,end+1:end+k) = uint32(0);
            end
            nlist(j,pos) = uint32(i);
            nsize_full(j) = pos;
        end
    end

    % remover duplicatas e zeros, e compactar
    maxdeg = 0;
    for i = 1:qtnode
        if nsize_full(i) == 0
            nsize(i) = uint16(0);
            continue;
        end
        row = nlist(i,1:nsize_full(i));
        row = row(row>0);              % tira zeros
        row = unique(row,'stable');    % tira duplicatas preservando ordem
        nsize(i) = uint16(numel(row));
        nlist(i,1:numel(row)) = row;
        maxdeg = max(maxdeg, numel(row));
    end

    % cortar colunas não usadas
    nlist = nlist(:,1:maxdeg);

end

function [nsize, nlist] = legacy_build_knn_with_labels(X, slabel, k, disttype)
%
% The original way of finding the k-nearest neighbors, as I wrote for [3]
%
% It is slower than the current version, but I kept it here just in case,
% since the neighbors lists produced by the newer method are not exactly
% the same, probably due to how knnsearch() handles ties differently from
% my implementation.
%
    qtnode = size(X,1);
    W = squareform(pdist(X,disttype).^2);  % gerando matriz de afinidade   
    % aumentando distâncias para todos os elementos, exceto entre os
    % rotulados de mesmo rótulo
    W = W + (~((repmat(slabel,[1,size(slabel)]) == repmat(slabel,[1,size(slabel)])') & (repmat(slabel,[1,size(slabel)])~=0)))*max(max(W));
    % eliminando a distância para o próprio elemento
    W = W + eye(qtnode)*realmax;       
    % construindo grafo
    graph = zeros(qtnode,'double');    
    for i=1:k-1
        [~,ind] = min(W,[],2);
        graph(sub2ind(size(graph),1:qtnode,ind')) = 1;
        graph(sub2ind(size(graph),ind',1:qtnode)) = 1;
        W(sub2ind(size(W),1:qtnode,ind')) = +Inf;
        %for j=1:qtnode
        %    graph(j,ind(j))=1;
        %    graph(ind(j),j)=1;
        %    W(j,ind(j))=+Inf;
        %end
    end
    % últimos vizinhos do grafo (não precisa atualizar W pq não será mais
    % usado)
    [~,ind] = min(W,[],2);
    clear W;
    graph(sub2ind(size(graph),1:qtnode,ind'))=1;
    graph(sub2ind(size(graph),ind',1:qtnode))=1;
    %for j=1:qtnode        
    %    graph(j,ind(j))=1;
    %    graph(ind(j),j)=1;
    %end
    clear ind;

    % criando célula para listas de vizinhos
    nsize = uint16(sum(graph));
    nlist = zeros(qtnode,max(nsize),'uint32');
    for i=1:qtnode       
        nlist(i,1:nsize(i)) = find(graph(i,:)==1);
    end
    clear graph
end