package Clustering;

import com.company.RetrievalAPI;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Clustering extends RetrievalAPI{
    private Map<String, Integer> termToIndex;  // maps the vocab term to corresponding column index in vector
    private List<INDArray> docClusters;   // INDArray at index i is the document term matrix in cluster i
    public static final String MIN_LINKAGE = "min";
    public static final String MAX_LINKAGE = "max";
    public static final String AVG_LINKAGE = "avg";
    public static final String MEAN_LINKAGE = "mean";
    private static int VOCAB_SIZE;
    private static List<Double> threshold;
    private static final double LOWER_THRESHOLD = 0.05;
    private static final double HIGHER_THRESHOLD = 0.95;
    private static final double THRESHOLD_STEP_SIZE = 0.10;

    public Map<String, Integer> getTermToIndex() {
        return termToIndex;
    }

    public void setTermToIndex(Map<String, Integer> termToIndex) {
        this.termToIndex = termToIndex;
    }

    public Clustering(boolean isCompressed) throws IOException {

        super(isCompressed);
        VOCAB_SIZE = super.getDiskReader().getRetrievedLookUpTable().size();
        System.out.println("Vocabulary Size : "+VOCAB_SIZE);
        buildTermToIndex();
        buildDocumentVector(1);

    }
    private void createThresholdList(){

        for(double i = LOWER_THRESHOLD; i<=HIGHER_THRESHOLD; i += THRESHOLD_STEP_SIZE){
            threshold.add(i);
        }

    }

    private void buildTermToIndex(){
        /* Takes the Term Look Up table and assigns an index sequentially.
        This index will be used for the vector.
         */
        termToIndex = new HashMap<>();
        int i = 1;  // start from 1 because 0 will be document id
        for(String term : super.getDiskReader().getRetrievedLookUpTable().keySet()){
            termToIndex.put(term,i++);
        }

    }

    private void addDocToCluster(int docId, String linkage) throws IOException{
        /*
        Inputs - Document Id, linkage choice(MIN, MAX, AVG, MEAN)
        Function : Takes the document Id and adds it to one of the clusters in the
        list of docClusters based on linkage choice
        Outputs - Void
         */
        int bestCluster = -1; // holds the index of the best cluster for this document. If -1,
                              // either the cluster list is empty or all of the clusters exceed threshold.
        double bestCost = Integer.MAX_VALUE;

        INDArray docVector = buildDocumentVector(docId);
        if(docVector == null){
            System.err.println("document Vector is NULL");
            System.exit(1);
        }
        double th = 0.05;


        for(int c  = 0; c<docClusters.size();c++){
            // For every cluster

            double cost = computeCost(docVector, docClusters.get(c), linkage);
            if(bestCost < th && bestCost>cost){
                bestCost = cost;
                bestCluster = c;
            }

        }









    }

    private double computeCost(INDArray docVector, INDArray cluster, String linkage){

        INDArray docVector_mod = docVector.get(NDArrayIndex.point(0), NDArrayIndex.interval(1,-1));

        INDArray cluster_mod = cluster.get(NDArrayIndex.all(), NDArrayIndex.interval(1,-1));
        INDArray costArray;

        double cost = 0;
            switch (linkage){
                case MAX_LINKAGE:
                    // Performs dot product and stores all the costs in |D| size column vector
                    costArray = cluster_mod.mmul(docVector.transpose());
                    cost = (double)costArray.maxNumber();
                    break;
                case MIN_LINKAGE:

                    costArray = cluster_mod.mmul(docVector.transpose());
                    cost = (double)costArray.minNumber();
                    break;
                case AVG_LINKAGE:
                case MEAN_LINKAGE:
                default:


            }

        return cost;

    }
    private INDArray buildDocumentVector(int docId) throws IOException{
        /*
        Inputs : document id
        Function : builds a 1 X (VOCAB SIZE + 1) vector. At index 0, the document ID is stored. This should be
        filtered out while performing operations.
        Output : Doc Term Vector
         */

        INDArray termVector = Nd4j.zeros(VOCAB_SIZE+1);
        termVector.put(0,0,docId);
        // Get the terms list for this document
        String[] docTerms = getDiskReader().getRetrievedDocToTermsMap().get(docId);
        // For each term, get its frequency in this doc and update term vector
        for(String term : docTerms){
            int fkid = super.getTermFrequencyInDoc(term,docId);
            int idx = termToIndex.get(term);
            termVector.put(0,idx, fkid);
        }

        return termVector;
    }
}

