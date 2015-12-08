/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann.credit.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

/**
 *
 * @author demorgan
 */
public class ANNCreditData {

    /**
     * @param args the command line arguments
     */
    static final Double Min_Credit_Amount = 250.0;
    static final Double Max_Credit_Amount = 18424.0;
    static final Double Min_Age = 19.0;
    static final Double Max_Age = 75.0;
    static final Double lrate = 0.9;
    static final Double momentum = 0.3;
    static final int Number_of_Epochs = 100;
    static final boolean Debug_Enable = false;

    static Double TP;
    static Double FP;
    static Double TN;
    static Double FN;

    static int asd = 0;

    static void outputs_init() {
        TP = 0.0;
        FP = 0.0;
        TN = 0.0;
        FN = 0.0;
    }

    public static void main(String[] args) throws FileNotFoundException, IOException {

        Double[] Enc_Credit_History = new Double[5];
        Double[] Enc_Employment = new Double[5];
        Double[] Enc_Property = new Double[4];
        Double[] Inputs = new Double[24];
        Double Credit_Amount = 0.0;
        Double Age = 0.0;
        Double[][] Input_Weights = new Double[24][8];
        Double[] Hidden_Weights = new Double[8];
        Double[] Hidden_Outputs = new Double[8];
        initilize_weights(Input_Weights, Hidden_Weights);
        Double ave_accuracy = 0.0;
        Double ave_tp_rate = 0.0;
        Double ave_tn_rate = 0.0;

        String line;
        String[] split;
        BufferedReader good_in = new BufferedReader(new FileReader("good.txt"));
        BufferedReader bad_in = new BufferedReader(new FileReader("bad.txt"));
        good_in.mark(100000);
        bad_in.mark(100000);
        Double prediction;
        Double output_error_rate = 0.0;
        Double actual;
        int in_read_good = 0;
        int out_read_good = 0;
        int in_read_bad = 0;
        int out_read_bad = 0;

        for (int f = 0; f < 10; f++) {
            for (int epo = 0; epo < Number_of_Epochs; epo++) {
                outputs_init();
                good_in.reset();
                bad_in.reset();
                in_read_good = 0;
                out_read_good = 0;
                in_read_bad = 0;
                out_read_bad = 0;
                for (int i = 0; i < 100; i++) {
                    if (i < (f * 10) || i >= (f + 1) * 10) {        //
                        for (int j = 0; j < 7; j++) {
                            //Good!
                            in_read_good++;
                            line = good_in.readLine();
                            split = line.split(",");
                            Enc_Credit_History = Encode_Credit_History(split[0]);
                            Credit_Amount = MinMax_Credit_Amount(Double.valueOf(split[1]));
                            Enc_Employment = Encode_Employment(split[2]);
                            Enc_Property = Encode_Property_Magnitude(split[3]);
                            Age = MinMax_Age(Double.valueOf(split[4]));
                            actual = 1.0;
                            Inputs = gather_inputs(Inputs, Enc_Credit_History, Credit_Amount, Enc_Employment, Enc_Property, Age);
                        
                            //overall
                            //go forward returns the double value of output. between 0-1. To be used in error calc.

                            prediction = go_forward(Inputs, Input_Weights, Hidden_Weights, Hidden_Outputs);
                            output_error_rate = get_output_error(prediction, actual);
                            Weight_Updates(Input_Weights, Hidden_Weights, Hidden_Outputs, output_error_rate, Inputs);
                        }
                        for (int j = 0; j < 3; j++) {
                            in_read_bad++;
                            //Bad!
                            line = bad_in.readLine();
                            split = line.split(",");
                            Enc_Credit_History = Encode_Credit_History(split[0]);
                            Credit_Amount = MinMax_Credit_Amount(Double.valueOf(split[1]));
                            Enc_Employment = Encode_Employment(split[2]);
                            Enc_Property = Encode_Property_Magnitude(split[3]);
                            Age = MinMax_Age(Double.valueOf(split[4]));
                            actual = 0.0;
                            Inputs = gather_inputs(Inputs, Enc_Credit_History, Credit_Amount, Enc_Employment, Enc_Property, Age);
                      
                        //overall
                            //go forward returns the double value of output. between 0-1. To be used in error calc.
                            prediction = go_forward(Inputs, Input_Weights, Hidden_Weights, Hidden_Outputs);
                            output_error_rate = get_output_error(prediction, actual);
                            Weight_Updates(Input_Weights, Hidden_Weights, Hidden_Outputs, output_error_rate, Inputs);
                        }
                    } else {
                        for (int j = 0; j < 7; j++) {
                            out_read_good++;
                            good_in.readLine();
                        }
                        for (int j = 0; j < 3; j++) {
                            out_read_bad++;
                            bad_in.readLine();
                        }
                    }
                }
            }
            if (Debug_Enable) {
                System.out.println("\n\ntrain in good " + in_read_good);
                System.out.println("train out good " + out_read_good);
                System.out.println("train in bad " + in_read_bad);
                System.out.println("train out bad " + out_read_bad);
            }
            good_in.reset();
            bad_in.reset();

            in_read_good = 0;
            out_read_good = 0;
            in_read_bad = 0;
            out_read_bad = 0;

            for (int i = 0; i < 100; i++) {            //TEST!
                if (i >= (f * 10) && i < (f + 1) * 10) {     //Good!
                    for (int j = 0; j < 7; j++) {
                        in_read_good++;                   
                        line = good_in.readLine();
                        split = line.split(",");
                        Enc_Credit_History = Encode_Credit_History(split[0]);
                        Credit_Amount = MinMax_Credit_Amount(Double.valueOf(split[1]));
                        Enc_Employment = Encode_Employment(split[2]);
                        Enc_Property = Encode_Property_Magnitude(split[3]);
                        Age = MinMax_Age(Double.valueOf(split[4]));
                        actual = 1.0;
                        Inputs = gather_inputs(Inputs, Enc_Credit_History, Credit_Amount, Enc_Employment, Enc_Property, Age);
                        
                        prediction = go_forward(Inputs, Input_Weights, Hidden_Weights, Hidden_Outputs);
                        test_output(prediction, actual);
                    }
                    for (int j = 0; j < 3; j++) {
                        //Bad!
                        in_read_bad++;
                        line = bad_in.readLine();
                        split = line.split(",");
                        Enc_Credit_History = Encode_Credit_History(split[0]);
                        Credit_Amount = MinMax_Credit_Amount(Double.valueOf(split[1]));
                        Enc_Employment = Encode_Employment(split[2]);
                        Enc_Property = Encode_Property_Magnitude(split[3]);
                        Age = MinMax_Age(Double.valueOf(split[4]));
                        actual = 0.0;
                        Inputs = gather_inputs(Inputs, Enc_Credit_History, Credit_Amount, Enc_Employment, Enc_Property, Age);

                        prediction = go_forward(Inputs, Input_Weights, Hidden_Weights, Hidden_Outputs);

                        test_output(prediction, actual);
                    }

                } else {
                    for (int j = 0; j < 7; j++) {
                        out_read_good++;
                        good_in.readLine();
                    }
                    for (int j = 0; j < 3; j++) {
                        out_read_bad++;
                        bad_in.readLine();
                    }
                }
            }
            if (Debug_Enable) {
                System.out.println("\n\nTest in good " + in_read_good);
                System.out.println("Test out good " + out_read_good);
                System.out.println("Test in bad " + in_read_bad);
                System.out.println("Test out bad " + out_read_bad);
            }
            System.out.println("Fold Number: " + f);
            System.out.println("TP: " + TP + " FN: " + FN + " FP: " + FP + " TN: " + TN);
            asd++;
            Double TP_Rate = TP / (TP + FN);
            ave_tp_rate += TP_Rate;

            Double Accuracy = (TP + TN) / (TP + TN + FP + FN);
            ave_accuracy += Accuracy;

            Double TN_Rate = TN / (TN + FP);
            ave_tn_rate += TN_Rate;

            System.out.println("True Positive Rate(sensivity) : " + TP_Rate);
            System.out.println("True Negative Rate(specificity) :" + TN_Rate);
            System.out.println("Accuracy:" + Accuracy);
        }
        ave_accuracy /= 10;
        ave_tn_rate /= 10;
        ave_tp_rate /= 10;
        System.out.println("\n");
        System.out.println("Average True Positive Rate(sensivity) : " + ave_tp_rate);
        System.out.println("Average True Negative Rate(specificity) :" + ave_tn_rate);
        System.out.println("Average Accuracy:" + ave_accuracy);
        good_in.close();
        bad_in.close();
    }

    static void test_output(Double predict, Double actual) {

        if (predict > 0.5) {
            predict = 1.0;
        } else {
            predict = 0.0;
        }

        if (actual == 1.0) { //good
            if (predict == 1.0) {
                TP++;    //actual= good & predict = good
            } else {
                FN++;                  //actual= good & predict = bad
            }
        } else {
            if (predict == 1.0) {           //actual= bad & predict = good
                FP++;
            } else {                      //actual= bad & predict = bad
                TN++;
            }
        }

    }

    static void Weight_Updates(Double[][] Input_Weights, Double[] Hidden_Weights, Double[] Hidden_Outputs, Double output_error_rate, Double[] Inputs) {
        Double[] Hidden_Error = new Double[8];
        NumberFormat formatter = new DecimalFormat("#0.00000");
        //System.out.println("\nUpdated Hidden Weights");
        for (int i = 0; i < 8; i++) {
            Hidden_Error[i] = Hidden_Outputs[i] * (1.0 - Hidden_Outputs[i]) * Hidden_Weights[i] * output_error_rate;
            Hidden_Weights[i] += lrate * momentum * output_error_rate * Hidden_Outputs[i];
            Hidden_Weights[i] = Double.valueOf(formatter.format(Hidden_Weights[i]));
            //System.out.print(i+":"+Hidden_Weights[i]+"| \t");
        }
        //System.out.println("Updated Input Weights");
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 24; j++) {
                Input_Weights[j][i] += momentum * lrate * Hidden_Error[i] * Inputs[j];
                //System.out.print(j+"->" + i +":" +Input_Weights[j][i]+"|" );
            }
            //System.out.println("");
        }

    }

    static Double get_output_error(Double prediction, Double actual) {
        return prediction * (1 - prediction) * (actual - prediction);
    }

    static Double go_forward(Double[] Inputs, Double[][] Input_Weights, Double[] Hidden_Weights, Double[] Hidden_Outputs) {
        double prediction;

        Double[] Hidden_Inputs = new Double[8];

        for (int i = 0; i < 8; i++) {
            Hidden_Inputs[i] = 0.0;
        }
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 24; j++) {
                Hidden_Inputs[i] += Inputs[j] * Input_Weights[j][i];
            }
        }
        Double the_output = 0.0;
        for (int i = 0; i < 8; i++) {
            Hidden_Outputs[i] = Sigmoid(Hidden_Inputs[i]);              //Activation Function
            the_output += Hidden_Outputs[i] * Hidden_Weights[i];        //Sum all the hidden layer outputs.
        }

        for (int i = 0; i < 8; i++) {
            Inputs[16 + i] = Hidden_Outputs[i];       // Thanks to the Elman Recurrent Neural Network!!
        }
        the_output = Sigmoid(the_output);

        prediction = the_output;

        return prediction;
    }

    static double Sigmoid(double x) {

        double result = 1.0 / (1.0 + Math.pow(Math.E, -x));
        // NumberFormat formatter = new DecimalFormat("#0.00000");     //yavaşlatmaya gerek yok şimdilik
        // result = Double.valueOf(formatter.format(result));
        return result;
    }

    static void initilize_weights(Double[][] Input_Weigths, Double[] Hidden_Weights) {
        Random a = new Random();

        NumberFormat formatter = new DecimalFormat("#0.0000");
        for (int i = 0; i < 8; i++) {
            Hidden_Weights[i] = Double.valueOf(formatter.format(a.nextDouble() /* - 0.5*/));    //get value between -0.5 <-> 0.5

        }
        for (int i = 0; i < 24; i++) {
            for (int j = 0; j < 8; j++) {
                Input_Weigths[i][j] = Double.valueOf(formatter.format(a.nextDouble() /* - 0.5*/));
            }
        }
    }

    static void divide_inputs() throws FileNotFoundException, UnsupportedEncodingException, IOException {

        BufferedReader in = new BufferedReader(new FileReader("part2-credit.txt"));
        String line = in.readLine();
        String[] split;
        PrintWriter good_writer = new PrintWriter("good.txt", "UTF-8");
        PrintWriter bad_writer = new PrintWriter("bad.txt", "UTF-8");
        while ((line = in.readLine()) != null) {
            split = line.split(",");
            if (split[5].equals("good")) {
                good_writer.println(line);
            } else if (split[5].equals("bad")) {
                bad_writer.println(line);
            } else {
                System.out.println("Erorr!");
            }
        }

        bad_writer.close();
        good_writer.close();
    }

    static Double[] gather_inputs(Double[] Inputs, Double[] Enc_Credit_History,
            Double Credit_Amount, Double[] Enc_Employment, Double[] Enc_Property, Double Age) {
        //first 5 nodes Credit History
        for (int i = 0; i < 5; i++) {
            Inputs[i] = Enc_Credit_History[i];
        }
        Inputs[5] = Credit_Amount;

        for (int i = 6; i < 11; i++) {
            Inputs[i] = Enc_Employment[i - 6];
        }
        for (int i = 11; i < 15; i++) {
            Inputs[i] = Enc_Property[i - 11];
        }
        Inputs[15] = Age;

        for (int i = 16; i < 24; i++) {
            Inputs[i] = 0.5;
        }

        return Inputs;

    }

    static Double MinMax_Age(Double Age) {
        NumberFormat formatter = new DecimalFormat("#0.000");
        Age = (Age - Min_Age) / (Max_Age - Min_Age);
        Age = Double.valueOf(formatter.format(Age));
        return Age;
    }

    static Double[] Encode_Property_Magnitude(String Property) {
        Double[] encoded = new Double[4];
        for (int i = 0; i < 4; i++) {
            encoded[i] = 0.0;
        }
        switch (Property) {
            case "'no known property'":
                encoded[0] = 1.0;
                break;
            case "car":
                encoded[1] = 1.0;
                break;
            case "'life insurance'":
                encoded[2] = 1.0;
                break;
            case "'real estate'":
                encoded[3] = 1.0;
                break;
            default:
                System.out.println("Property error!" + Property);
                break;
        }
        return encoded;
    }

    static Double[] Encode_Credit_History(String Credit) {
        Double[] encoded = new Double[5];
        for (int i = 0; i < 5; i++) {
            encoded[i] = 0.0;
        }
        switch (Credit) {
            case "'critical/other existing credit'":
                encoded[0] = 1.0;
                break;
            case "'existing paid'":
                encoded[1] = 1.0;
                break;
            case "'delayed previously'":
                encoded[2] = 1.0;
                break;
            case "'no credits/all paid'":
                encoded[3] = 1.0;
                break;
            case "'all paid'":
                encoded[4] = 1.0;
                break;
            default:
                System.out.println("Credit error!" + Credit);
                break;
        }
        return encoded;
    }

    static Double MinMax_Credit_Amount(Double Amount) {
        NumberFormat formatter = new DecimalFormat("#0.00000");
        Amount = (Amount - Min_Credit_Amount) / 18174.0;
        Amount = Double.valueOf(formatter.format(Amount));
        return Amount;
    }

    static Double[] Encode_Employment(String Employment) {
        Double[] encoded = new Double[5];
        for (int i = 0; i < 5; i++) {
            encoded[i] = 0.0;
        }
        switch (Employment) {
            case "unemployed":
                encoded[0] = 1.0;
                break;
            case "1<=X<4":
                encoded[1] = 1.0;
                break;
            case ">=7":
                encoded[2] = 1.0;
                break;
            case "<1":
                encoded[3] = 1.0;
                break;
            case "4<=X<7":
                encoded[4] = 1.0;
                break;
            default:
                System.out.println("Employment error!" + Employment);
                break;
        }
        return encoded;

    }

}
