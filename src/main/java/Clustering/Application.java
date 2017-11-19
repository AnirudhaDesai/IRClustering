package Clustering;

import org.apache.log4j.BasicConfigurator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;


public class Application {

    public static void main( String[] args ) throws IOException
    {
        BasicConfigurator.configure();

          /* Parse input for compressed option :
         "c"   -> Compressed
         default -> Uncompressed
         */



        boolean isCompressed = false;
        int numOfResults = 5;    // Default number of retrieved documents


        if(args.length != 0 && (args[0].equals("c") || args[0].equals("C"))) {
            isCompressed = true;
            System.out.println("Running in Mode : Compressed");
        }
        else
            System.out.println("Running in Mode : Uncompressed");
        String mode = isCompressed==false?"Uncompressed":"Compressed";

        Clustering clustering = new Clustering(isCompressed);


        INDArray A = Nd4j.ones(2,2);
        INDArray B = Nd4j.ones(1,2);
        B.put(0,0,5);
        System.out.println("B Min : "+B.amin(1));
        INDArray prod = A.mmul(B.transpose());
        System.out.println(prod);




    }
}
