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
% k         - Each node is connected to its k-neirest neighbors. Default:
%             size of the dataset multiplied by 0.05.
% disttype  - Use 'euclidean', 'seuclidean', etc. Default: 'euclidean'.
%             See the MATLAB knnsearch funcion help for all the options.
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
% mex       - Uses the mex version of the code (compiled binary) which is
%             ~10 times faster. Default: true. Set to false to use the 
%             Matlab only version.
% useseed   - Set to true if you want to use a seed to allow reproducible
%             results. Default: false. Remember to set the seed with the
%             seed option.
% seed      - Seed to itialize the random number generator, which is rng()
%             or rand_s() in non-mex and mex versions, respectively. 
%             Default: 0. Remember to set the option useseed to true
%             if you want the provided seed to be used. 
% Xgraph    - Set to true if X is a pre-built graph instead of a feature
%             matrix.
% legacy_knn- Default: false. When set to 'true',it uses the original form
%             of finding the k-nearest neighbors, which is slower in large
%             datasets. 
%
% OUTPUT:
% owner     - vector of classes assigned to each data item
% pot       - final ownership level of each item. Warning: it should not
%             interpreted as pertinence level, use owndeg instead.
% owndeg    - fuzzy output as in [2], each line is a data item, each column
%             pertinence to a class
% distnode  - matrix with the distance vectors of each particle, each
%             column is a particle and each line is a node
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
        %options.mex logical = true % uses the mex version
        options.useseed logical = false % do not set seed
        options.seed int32 = 0 % random seed
        options.Xgraph = false % X is a feature matrix
        options.legacy_knn = false % legacy function to find k-nearest neighbors
    end   

    qtnode = size(X,1);

    % tratamento da entrada
    slabel = uint16(slabel);
    k = uint16(options.k);   
    % constantes
    potmax = 1.000; % potencial máximo
    potmin = 0.000; % potencial mínimo
    npart = sum(slabel~=0); % quantidade de partículas
    %stopmax = round((qtnode/npart)*round(options.valpha*0.01)); % qtde de iterações para verificar convergência
    stopmax = round((qtnode/double(npart*k))*round(options.valpha*0.1));

    if(~options.legacy_knn)
        [nsize, nlist] = build_knn_with_labels(X, slabel, k, options.disttype);        
    else
        [nsize, nlist] = legacy_build_knn_with_labels(X, slabel, k, options.disttype);
    end

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
        % initializing the accumlated dominance vectors
        % we can't use 0, otherwise non-visited nodes would trigger a division
        % by zero.
        owndeg = repmat(realmin,qtnode,options.nclass);  

        lnpccloop(options.maxiter, npart, options.nclass, options.earlystop, ...
            stopmax, options.pgrd, options.dexp, options.deltav, ...
            options.deltap, potmin, partpos, partclass, potpart, ...
            slabel, nsize, distnode, nlist, pot, owndeg, ...
            options.useseed, options.seed);

        potacc = potacc + pot;
    end
    [~,owner] = max(potacc,[],2);
    owndeg = owndeg ./ repmat(sum(owndeg,2),1,options.nclass);
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