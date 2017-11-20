package Clustering;

import com.company.RetrievalAPI;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.IOException;
import java.util.*;

public class Clustering extends RetrievalAPI{
    private Map<String, Integer> termToIndex;  // maps the vocab term to corresponding column index in vector
    private List<INDArray> docClusters;   // INDArray at index i is the document term matrix in cluster i
    List<List<Integer>> clusters;
    private static final String MIN_LINKAGE = "min";
    private static final String MAX_LINKAGE = "max";
    private static final String AVG_LINKAGE = "avg";
    private static final String MEAN_LINKAGE = "mean";
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
        docClusters = new ArrayList<>();
        System.out.println("Vocabulary Size : "+VOCAB_SIZE);
        buildTermToIndex();
        runClustering(MIN_LINKAGE, 0.30);
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
    public List<List<Integer>> runClustering(String linkage, double threshold){
        /*
        Input - linkage choice(MIN, MAX, AVG, MEAN), threshold
        Function  : Runs the clustering process on all of the documents in the index
        Output - List of Clusters. Each cluster is in turn a list of document numbers.
         */
        List<Integer> docIdList = new ArrayList<>();
        clusters = new ArrayList<>();

        Set<Integer> allDocs = super.getDiskReader().getRetrievedDocToLengthMap().keySet();
        // Stream every doc to the clustering process
        int ik = 100;
        for(Integer doc : allDocs){
            try {
                addDocToCluster(doc, linkage, threshold);
            }catch (IOException e){
                e.printStackTrace();
            }
            ik--;
            if(ik<=0) break;

        }
        // All documents are now clustered but are in INDArray matrices. Retrieve and group document numbers
        System.out.println("Number of Clusters : "+docClusters.size());
        for(int i = 0 ; i<docClusters.size();i++ ){
            INDArray docs = docClusters.get(i).getColumn(0);
            docIdList = new ArrayList<>();

            for(int j = 0; j<docs.rows();j++)
                docIdList.add(docs.getInt(j));
            clusters.add(docIdList);

        }

        return clusters;
    }

    private void addDocToCluster(int docId, String linkage, double threshold) throws IOException{
        /*
        Inputs - Document Id, linkage choice(MIN, MAX, AVG, MEAN), threshold
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
            System.out.println(cost);
            if(cost < threshold && bestCost>cost){
                bestCost = cost;
                bestCluster = c;
            }

        }

        if(bestCluster == -1){
            // Either the doc cluster is empty or the cost is higher than threshold with all clusters
          docClusters.add(docVector);
        }
        else{
            //  Appending the doc to the cluster matrix at index bestCluster
            Nd4j.vstack(docClusters.get(bestCluster), docVector);
        }

    }

    private double computeCost(INDArray docVector, INDArray cluster, String linkage){

        INDArray docVector_mod = docVector.get(NDArrayIndex.point(0), NDArrayIndex.interval(1,docVector.columns()));

        INDArray cluster_mod = cluster.get(NDArrayIndex.all(), NDArrayIndex.interval(1,cluster.columns()));
        List<Double> costArray = new ArrayList<>();

        double cost = 0;
            switch (linkage){
                case MIN_LINKAGE:
                    // Find the cosine similarity
                    for(int i = 0;i<cluster_mod.rows();i++)
                        costArray.add(Transforms.cosineDistance(cluster_mod.getRow(i), docVector_mod));
                    cost = Collections.min(costArray);
                    break;
                case MAX_LINKAGE:
                    for(int i = 0;i<cluster_mod.rows();i++)
                         costArray.add(Transforms.cosineDistance(cluster_mod.getRow(i), docVector_mod));
                    cost = Collections.max(costArray);
                    break;
                case AVG_LINKAGE:
                    for(int i = 0;i<cluster_mod.rows();i++)
                        costArray.add(Transforms.cosineDistance(cluster_mod.getRow(i), docVector_mod));
                    // Average all the costs
                    double sum = 0;
                    for(double c : costArray)
                        sum += c;
                    cost = sum/(double)costArray.size();
                    break;

                case MEAN_LINKAGE:


                default:
                    break;
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

