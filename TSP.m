function bestTourOverall = TSP(szCSVfilename)
%TRAVLELLING SALESMAN PROBLEM
%
% usage: bestTour = TSP(szCSVfilename)

%% 1. Write a program that will produce the most efficient cycle of visits for the salesman. You should optimise for Cost, Travel Time and Both Cost & Travel Time.
%
% The Travelling Salesman Problem (TSP) is a well known NP-hard combinatorial
% optimization problem.
%
% Producing the most efficient (optimal) tour, although possible for
% problems of this size, was not feasible in the timescale. (In terms
% of development time available) I have therefore produced a suboptimal
% implementation derived from some of the (less sophisticated) Travelling
% Salesman Problem algorithms & heuristics available from on-line resources
% and the literature.
%
% To further simplify the problem, I formulated my assumptions such that
% the cost of hotels was equal for all possible tours, meaning that only an
% optimisation for travel time was required.
%
% I succeeded in implementing a number of simple tour construction
% algorithms, and attempted (and failed) to implement a 2-opt tour
% improvement algorithm.

%% 2. Would you use your algorithm if all problems you faced had only three cities? What about 10? 100? how about 15000?
%
% The number of routes increases as the factorial of the number of cities.
% A brute force search will therefore be preferable for 3 cities, (6
% possible routes), and might be acceptable for 10 cities (3,628,800 possible
% routes) but will quickly stop being practicable as the number of cities
% increases beyond 10.
%
% The tour construction algorithm that I have implemented has a computational
% complexity of maybe Oh-cubed (I am not sure), so it is suitable
% for problems with a few hundred cities. For problems with of around 15000
% cities in size, it would be appropriate to spend more time optimising
% the algorithm, for example, by restricting the search for vertex insertion
% locations to edges connected to the k nearest neighbours of the vertex
% being inserted.
%

%% 3. How did you chose the algorithm that you used? Why did you chose that particular approach.
%
% I took an iterative, rapid-prototyping approach to development,
% implementing simpler algorithms first, and then improving upon them,
% concurrently searching through the literature and other online resources
% for guidance. I did this to minimise development risk, and to facilitate
% the use of development effort to support learning activities.
%
% My initial thoughts centred around the optimisation aspects of the
% problem; in particular, I identified the need to get a good starting point
% close to the global minimum (or a set of starting points with at least one
% close to the global minimum), as well as the need to get an effective
% algorithm for choosing good successor points. I was (and am still) unclear
% of the best definition of "closeness" or similarity between tours, and the
% impact of that definition on the choice of successor-candidate-generation
% algorithm.
%
% A quick initial plot of the data showed a high degree of clustering,
% suggesting that a hierarchical / decomposition approach to the solution
% might be advantageous (although this avenue was not explored any further).
%
% Additionally, the histogram of edge lengths indicated that it might be
% possible to represent edge distances by uint8 integers with only limited
% levels of saturation (although a 32 bit fixed point representation was
% eventually chosen).
%
% I spent a little bit of time considering the Delaunay triangulation as
% a source of the initial tour, as well as trying to think of how the
% notion of the gradient of the objective function translated into the
% combinatorial optimisation problem domain. (These considerations turned
% out to be red herrings).
%
% My first implementation (to get started with) was a very naive, with
% the starting tour generated with randperm, and an update step that
% simply swapped n vertices to produce n successor tours, with the best
% of these carried forwards to the next iteration. This was little more
% than a dummy implementation around which to build the structure of the
% program.
%
% The second implementation was the first that produced anything like
% a reasonable tour. It was based on iteratively adding points to a
% convex hull, as suggested by an online reference. Initially, I added
% points only to the end of the tour, and then improved upon this
% by adding a point insertion routine. I tried both nearest-neighbour
% and farthest-point heuristics with these approaches, measuring the
% results, and found the farthest-point heuristic to give consistently
% better results.
%
% I then simplified the source a little by removing the convex hull
% initialisation routine, building the tour from a single "seed" point.
% I also tried a number of different heuristics, including, amongst others,
% most-isolated-point first, random point selection, and farthest-point
% selection. These all produced better-than-random tours, although none
% produced tours that were better than the farthest-point heuristic.
%
% After this, I moved on to the incremental updating routine. Having
% read a little of the literature by this point, I decided that a 2-opt
% like updating algorithm seemed like a logical next step up in complexity,
% but my first attempt failed, tending to make the tour worse, rather than
% better.
%
% At this point, I ran out of time, just as I was considering strategies
% to identify 2-opt moves with a higher probability of improving the
% tour. (limiting moves to k-nearest-neighbours), so I removed the 2-opt
% functionality, tidied the code for presentation and completed this
% narrative.
%
% Terminology:
%
% A "Vertex" in the graph corresponds to a City in the problem statement.
% An "Edge" in the graph corresponds to a journey between two cities.
% A "Tour" corresponds to an ordering of the vertices in the problem set.
%
%

%% 4. How do you know that you code works?
%
% Each tour that is produced is validated using an assertion
% like construct. No unit tests have been produced, no
% static analysis has been performed, and no independent
% review has been undertaken, so the level of assurance is
% minimal.

%% 5. How do you know that your answer is "reasonable"? What does "reasonable"? mean in this case? Is your solution "optimal"? Is it the "best" possible solution?
%
% The term "reasonable" indicates that a trade-off has been
% made between development time/risk, run-time performance and
% "optimality" of the results produced by the algorithm.
% It was known that development time was limited, that few risks
% could be taken (for example, on the development of novel
% algorithms), and that the experience of the developer was limited
% (in combinatorial optimisation problems). It was also known that
% feasible solutions that provide "optimal" tours would be complex
% both in terms of development time and run time. As a result,
% many simple algorithms could be seen as "reasonable"
% implementations.
%

%% 6. What is the time/space complexity of your algorithm
%
% The space complexity of the tour construction algorithm is
% about Oh-squared, as it makes use of several matrices
% nVertices-squared in size.
%
% The time complexity of the (unoptimised) tour construction
% algorithm is probably around Oh-cubed in size, as for each
% point that gets added to the tour, a tourSize-squared
% matrix is processed.
%

%% 7. How would you parallelise your solution?
%
% The basic tour construction algorithm (growing a single tour)
% cannot be parallelised without, in all likelihood, having to
% significantly restructure it. (E.g. to break the problem down
% into sub-problems). The construction of a starting "population"
% of n tours can be parallelised much more easily, with ceil(n/k)
% tours per worker for k workers.
%
% An iterative tour improvement (optimisation) algorithm (not
% yet implemented) could probably be parallelised fairly easy
% by searching at several points simultaneously, in the manner
% of a genetic algorithm.
%

%% 8. How would you improve your algorithm if you had more time?
%
% Firstly, I would try to get a 2-opt tour improvement algorithm
% working, by implementing (for example) optimisations to limit
% the number of vertices processed to the k-nearest neighbours
% of the vertex under examination. This should provide a reasonable
% "second cut" solution without too much development effort or
% risk.
%
% After getting a basic 2-opt tour improvement algorithm working,
% I would probably look at some of the techniques implemented
% by Lin & Kernighan, and try to find any that translate nicely
% into MATLAB. (For example, the marking of already visited nodes
% by a boolean array).
%
% Next, I would try to implement some of the concepts from "Genetic
% Algorithms", specifically because the representation of a population
% of tours as a matrix in MATLAB is quite tempting.
%
% Finally, and given enough time, it would also be tempting to try
% to implement more advanced optimisation control strategies,
% such as "Simulated Annealing", to try to understand the problem
% in a more academic way.

%% Assumptions:
%
%  1. Each city must be visited once and only once; even the start city
%     and the end city must be different. (The tour is "open").
%  2. The end-points are not fixed; the tour may start and end at any city.
%  3. The travel distance is the euclidean distance between cities, and
%     the travel time is proportional (and equal) to the travel distance.
%  4. Travel is free, and the only cost is that of staying in hotels.
%  5. The analyst spends exactly one night in each city including
%     both the cities at the start and the end of the tour.
%  8. An exact solution is not required.
%  9. The algorithm must be fully automatic. Human supervision/input
%     is not permitted.
%
% NOTE: The effect of 1,4 & 5 is that the "cost" is invariant over
%       all possible tours. We will therefore only solve for minimum
%       travel time.
%

   if 0==nargin
      szCSVfilename = which('TSP.csv');
   end

   [costTable,proxSeqTable,vertexX,vertexY] = TSP_loadProblem( szCSVfilename );

   % Tune this value to control the running-time
   nTours = uint32(4);
   tours  = TSP_createInitialTours( vertexX, vertexY, costTable, nTours );

   % Record & display the best tour found so far.
   tourCosts = TSP_objective( costTable, tours );
   [bestCost,iBest] = min(tourCosts);

   bestTourOverall = tours(iBest,:);
   bestCostOverall = bestCost;

   TSP_plotTour( vertexX, vertexY, bestTourOverall );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tours = TSP_createInitialTours( vertexX, vertexY, costTable, nTours )
%TSP_createInitialTours - creates the initial population of tours
%
% This function generates an initial "population" of tours using a
% one of a selection of tour construction algorithms.
%

   nVertices = uint32(numel(vertexX));
   tours     = zeros( nTours, nVertices, 'uint32' );

   % We could profitably farm the operations in this loop out over a
   % distributed computing cluster, as this is an "embarrasingly"
   % parallell problem.

   vrtxSelectHeurstc = 'farthest';
   for iTour = 1:nTours

STDOUT = 1;
fprintf( STDOUT, 'Generating tour: %04.0f / %04.0f\n', iTour, nTours );

      tours( iTour, : ) = TSP_growTour( vertexX, vertexY, costTable, vrtxSelectHeurstc );

      if    numel(vertexX) ~= numel(tours(iTour,:))         ...
         || numel(vertexX) ~= numel(unique(tours(iTour,:)))
         error( 'TSP:TSP', 'Tour is not a valid tour.' );
      end

   end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tour = TSP_growTour( vertexX, vertexY, costTable, vrtxSelectHeurstc )
%TSP_growTour - grow tour using various heuristics.
%
% The tour is grown from a single vertex. On each iteration, a
% single unassigned vertex is identified using one from a selection of
% heuristics, and that vertex is added to the tour at the optimal
% insertion point.

   nVertices = uint32(numel(vertexX));
   tour      = zeros( 1, nVertices, 'uint32' );
   isInTour  = false( 1, nVertices );
   tourSize  = uint32(1);

   % Build some de-referencing tables for different orderings of vertices
   % (Used by some of the heuristics below).
   tempCostTable                   = costTable;
   tempCostTable(0==tempCostTable) = nan;
   [unused,isolatedFirstOrdering]  = sort(min(tempCostTable,[],1),'descend'); %#ok<ASGLU>
   isolatedFirstOrdering           = uint32(isolatedFirstOrdering(:));

   randomOrdering                  = randperm( nVertices );
   randomOrdering                  = uint32(randomOrdering(:));

   % Select the first vertex to use. If a first vertex is not specified
   % by the chosen heuristic, pick one at random.
   switch vrtxSelectHeurstc
      case 'isolatedFirst'
         iVertexToAdd = isolatedFirstOrdering(1); % Select the first vertex in the tour.
      otherwise
         iVertexToAdd = randomOrdering(1);
   end

   isInTour( iVertexToAdd ) = true;
   tour( tourSize ) = iVertexToAdd;

   for tourSize = 2:nVertices

      switch vrtxSelectHeurstc

         case 'farthest'
         % Select vertex farthest from tour (Similar to the farthest
         % insertion heuristic from:
         %    http://www-e.uni-magdeburg.de/mertens/TSP/node2.html)
         %
         % This seems to be the best performing of the vertex selection
         % heuristics that I have tried so far.

            distToNearPoint = nan( 1, nVertices );
            distToNearPoint(~isInTour) = min( costTable(isInTour,~isInTour), [], 1 );
            [unused,iVertexToAdd] = max( distToNearPoint(:) );  %#ok<ASGLU>

         case 'random'
         % Select vertices to add at random. Surprisingly, this seems to be
         % quite an effective heuristic, close to the farthest-insertion
         % heuristic above.

            iVertexToAdd = randomOrdering( tourSize );

         case 'isolatedFirst'
         % Select vertices in order of isolation, the most isolated vertices
         % first This did not perform too badly either.

            iVertexToAdd = isolatedFirstOrdering( tourSize );

         case 'NNinsert'
         % Select the vertex closest to the tour. This is similar to the
         % Nearest Neighbour heuristic, but with insertion. It does not
         % seem to produce a particularly good initial tour.

            distToNearPoint = nan( 1, nVertices );
            distToNearPoint(~isInTour) = min( costTable(isInTour,~isInTour), [], 1 );
            [unused,iVertexToAdd] = min( distToNearPoint(:) );  %#ok<ASGLU>

         otherwise
            error( 'TSP:TSP_growTour', 'Unrecognised tour construction heuristic.' );

      end

      % Select the optimal insertion point in the tour to add the vertex;
      % We do this by evaluating every possible insertion point.
      % A tourSize-by-tourSize matrix is formed with each row holding a tour
      % for a single insertion point. The various possible insertion
      % points trace out the main diagonal of this matrix. Portions of
      % the tours prior to the insertion point form the lower triangular (LT)
      % part of this matrix; portions following the insertion point form
      % the uppert triangular (UT) part of this matrix.

      % TODO: 1. Investigate use of "toeplitz" function to construct
      %          the candidates matrix.
      %       2. Investigate use of the "Proximity Sequence Table" to
      %          limit the size of the candidates matrix. (I.e. only
      %          try inserting near kth nearest neighbours).

      temp = repmat( tour(1:tourSize-1), [tourSize-1,1] );

      candidatesLT                = zeros( tourSize, tourSize, 'uint32' );
      candidatesLT(2:end,1:end-1) = temp;
      candidatesLT                = tril(candidatesLT,-1);

      candidatesUT                = zeros( tourSize, tourSize, 'uint32' );
      candidatesUT(1:end-1,2:end) = temp;
      candidatesUT                = triu(candidatesUT,1);

      candidates = candidatesLT;
      isUT = (candidatesUT>0);
      candidates( isUT ) = candidatesUT( isUT );

      isDiag = (candidates==0);
      candidates( isDiag ) = iVertexToAdd;

      candidateCosts = TSP_objective( costTable, candidates );
      [unused,iBest] = min(candidateCosts(:)); %#ok<ASGLU>

      % Update the tour with the "best" tour.
      isInTour(iVertexToAdd) = true;
      tour(1:tourSize) = candidates(iBest,:);

   end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [costTable,proxSeqTable,vertexX,vertexY] = TSP_loadProblem( szCSVfilename )

   if    ~ischar( szCSVfilename ) ...
      || ~exist( szCSVfilename, 'file' )
      error( 'TSP:TSP_loadProblem', 'Invalid arument, filename not recognised');
   end

   % Read the CSV file, and give names to the columns.
   CSVdata     = csvread( szCSVfilename, 1, 0 );
   nVertices   = size( CSVdata, 1 );
   vertexX     = CSVdata( :, 2 );
   vertexY     = CSVdata( :, 3 );
   vertexCosts = CSVdata( :, 4 );

   % Calculate a Look-Up-Table of costs.
   % (Makes the objective function computationally trivial to evaluate).
   vertexXtable = repmat( vertexX(:),     [1,nVertices] );
   vertexYtable = repmat( vertexY(:),     [1,nVertices] );
   vertexCosts  = repmat( vertexCosts(:), [1,nVertices] ); %#ok<NASGU> - because hotel costs will be the same for all possible tours.

   euclideanDist = sqrt(   (vertexXtable-vertexXtable').^2 ...
                         + (vertexYtable-vertexYtable').^2 );

   % Remap the euclidean distance to an integer fixed-point representation,
   % with as much precision as we can fit into 32 bit unsigned integers.
   % (Integer operations on the cost table should be MUCH more efficient
   % than floating point operations). This is worth doing because cost
   % table operations are called a lot in the inner loops of almost all
   % TSP algorithms.

   minDist     = min(euclideanDist(euclideanDist>0));
   maxDist     = max(euclideanDist(:));
   distRange   = maxDist - minDist;
   targetRange = double(intmax('uint32')) / nVertices;
   fixedPtDist = uint32(((euclideanDist - minDist) / distRange) * targetRange);

   % The costTable is derived from euclidean distance only. Hotel costs
   % will be the same for all tours, so we can ignore them in our objective
   % function.
   costTable = fixedPtDist;

   % A "Proximity Sequence Table" is also calculated, to assist with
   % quickly finding k-nearest neighbours etc... The ith row of the
   % "Proximity Sequence Table" contains a sequence of vertex indices,
   % in order of proximity, from nearest to farthest, from the ith vertex.
   [unused,proxSeqTable] = sort(costTable,2); %#ok<ASGLU>


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tourCosts = TSP_objective( costTable, tours )
% TSPobjective evaluates one or more tour sequences and returns the cost.
%
% Tours are specified as a sequence of vertex indices. Each tour takes
% one row of the tours matrix. Multiple tours may be specified, but
% all tours must be the same length.

   edgeStartPoints = tours( :, 1:end-1 );
   edgeEndPoints   = tours( :, 2:end   );
   costTableHeight = uint32(size(costTable,1));

   % The algorithm spends more time evaluating the following statement than
   % any other. Since tours often remain largely unchanged from evaluation
   % to evaluation, we might want to consider moving as much of this
   % computation outside of the inner loop as possible, modifying iEdges
   % incrementally rather than recalculating it in its entirety each time
   % TSP_objective is called.

   iEdges    =   edgeStartPoints ...
               + ((edgeEndPoints-1) * costTableHeight);

   edgeCosts = costTable( iEdges );
   tourCosts = sum( edgeCosts, 2 );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function TSP_plotTour( vertexX, vertexY, tour )
%TSP_plotTour - visualisation function for TSP

   hFig = figure( 8484 );
   hAx  = gca;
   set( hFig, 'color', [1 1 1] );
   cla;
   set( hAx, 'position', [0.05 0.05 0.9 0.9] );
   hold( 'on' );

   plot( vertexX( : ), ...
         vertexY( : ), 'k.' );
   plot( vertexX( tour(~isnan(tour)) ), ...
         vertexY( tour(~isnan(tour)) ), 'bo-' );
   box(  'on' );
   grid( 'on' );
   axis( 'equal' );
   pause( 0.01 );

