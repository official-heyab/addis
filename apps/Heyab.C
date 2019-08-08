#include "ligra.h"
#include "math.h"
#include <queue>
#include <mutex>

//Global variables
const int vertexNum =100000;
const int queueNum =100;
int vertexPerQueue=vertexNum/queueNum;

queue<double> myQueue[queueNum];
mutex mtx[queueNum];


//Function which writes page rank to file
void pageRankToFile(double p_next[], int n){
  //Writing page rank value to heyab.txt
  ofstream myfile;
  myfile.open ("../inputs/heyab.txt");
  parallel_for(long i=0;i<n;i++) {
    myfile<<p_next[i]<<"\n";
  }
  myfile.close();
}

template <class vertex>
struct PR_F {
  double* p_curr, *p_next;
  vertex* V;
  PR_F(double* _p_curr, double* _p_next, vertex* _V) : 
    p_curr(_p_curr), p_next(_p_next), V(_V) {}
  inline bool update(uintE s, uintE d){ //update function applies PageRank equation
    p_next[d] += p_curr[s]/V[s].getOutDegree();
    return 1;
  }
  inline bool updateAtomic (uintE s, uintE d) { //atomic Update
    //writeAdd(&p_next[d],p_curr[s]/V[s].getOutDegree());
    
    //Getting queue index based on the destination id
    int queueIndex = d/vertexPerQueue;

    //Locking the specific mutex for that queue
    mtx[queueIndex].lock();

    //Adding the page rank + destination id to the queue
    myQueue[queueIndex].push(p_curr[s]/V[s].getOutDegree()+d);

    //Unlocking the specific mutex
    mtx[queueIndex].unlock();
   
    return 1;
  }
  inline bool cond (intT d) { return cond_true(d); }};

//vertex map function to update its p value according to PageRank equation
struct PR_Vertex_F {
  double damping;
  double addedConstant;
  double* p_curr;
  double* p_next;
  PR_Vertex_F(double* _p_curr, double* _p_next, double _damping, intE n) :
    p_curr(_p_curr), p_next(_p_next), 
    damping(_damping), addedConstant((1-_damping)*(1/(double)n)){}
  inline bool operator () (uintE i) {
    p_next[i] = damping*p_next[i] + addedConstant;
    return 1;
  }
};

//resets p
struct PR_Vertex_Reset {
  double* p_curr;
  PR_Vertex_Reset(double* _p_curr) :
    p_curr(_p_curr) {}
  inline bool operator () (uintE i) {
    p_curr[i] = 0.0;
    return 1;
  }
};

template <class vertex>
void Compute(graph<vertex>& GA, commandLine P) {
  long maxIters = P.getOptionLongValue("-maxiters",100);
  const intE n = GA.n;
  const double damping = 0.85, epsilon = 0.0000001;
  
  double one_over_n = 1/(double)n;
  double* p_curr = newA(double,n);
  {parallel_for(long i=0;i<n;i++) p_curr[i] = one_over_n;}
  double* p_next = newA(double,n);
  {parallel_for(long i=0;i<n;i++) p_next[i] = 0;} //0 if unchanged
  bool* frontier = newA(bool,n);
  {parallel_for(long i=0;i<n;i++) frontier[i] = 1;}

  vertexSubset Frontier(n,n,frontier);
  
  long iter = 0;
  while(iter++ < maxIters) {
    edgeMap(GA,Frontier,PR_F<vertex>(p_curr,p_next,GA.V),0, no_output);
    vertexMap(Frontier,PR_Vertex_F(p_curr,p_next,damping,n));

    //Using parallel loop adding the page rank of the respective 
    //Destination index found inside the queue
    parallel_for(long i=0;i<queueNum;i++) {
      while (!myQueue[i].empty()){ 
        //By getting the floor value, we know the destination index     
        int d=floor(myQueue[i].front()); 

        //Subtracting the destination index from the queue value will
        //give us page rank
        p_next[d] += myQueue[i].front()-d;
        myQueue[i].pop();       
      }      
    }

    //Writing the page rank to file for debugging purposes
    pageRankToFile(p_next, n);  

    //compute L1-norm between p_curr and p_next
    {parallel_for(long i=0;i<n;i++) {
      p_curr[i] = fabs(p_curr[i]-p_next[i]);
      }}
    double L1_norm = sequence::plusReduce(p_curr,n);
    if(L1_norm < epsilon) break;
    //reset p_curr
    vertexMap(Frontier,PR_Vertex_Reset(p_curr));
    swap(p_curr,p_next);
  }
  Frontier.del(); free(p_curr); free(p_next);     
}

