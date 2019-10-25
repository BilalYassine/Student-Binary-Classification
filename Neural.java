
// Title:           Neural.Java
// Purpose:         A program which emulates a neural network to perform binary classification.
// Author:          Bilal Yassine

import java.util.ArrayList;

import java.io.File;

import java.util.Scanner;

import java.io.FileNotFoundException;

class Record {

    public double[] x;

    public double y;

    public Record(double x1, double x2, double y) {
        x = new double[2];
        x[0] = x1;
        x[1] = x2;
        this.y = y;
    }

}

public class Neural {

    private static double[] array;

    private static final String TRAIN = "hw2_midterm_A_train.txt";

    private static final String EVAL = "hw2_midterm_A_eval.txt";

    private static final String TEST = "hw2_midterm_A_test.txt";

    public static ArrayList<Record> readFromFile(String filename) {
        ArrayList<Record> records = new ArrayList<>();
        File file = new File(filename);
        try (Scanner fin = new Scanner(file)) {
            while (fin.hasNextLine()) {
                String line = fin.nextLine();
                String[] tokens = line.split("\\s+");
                double x1 = Double.parseDouble(tokens[0]);
                double x2 = Double.parseDouble(tokens[1]);
                double y = Double.parseDouble(tokens[2]);
                records.add(new Record(x1, x2, y));
            }
        } catch (FileNotFoundException e) {
            System.out.println("File " + filename + " is not found.");
            System.exit(1);
        }
        return records;
    }

    private static double sigmoid(double z) {
        return 1.0 / (1 + Math.exp(-z));
    }

    private static double rectified(double z) {
        return Math.max(z, 0);
    }

    private static double[] eval(double[] x) {
        double uA = array[0] + array[1] * x[0] + array[2] * x[1];
        double vA = rectified(uA);
        double uB = array[3] + array[4] * x[0] + array[5] * x[1];
        double vB = rectified(uB);
        double uC = array[6] + array[7] * vA + array[8] * vB;
        double vC = sigmoid(uC);
        return new double[] {uA, vA, uB, vB, uC, vC};
    }

    private static double[] outputLayer(double[] values, double y) {
        double vC = values[5];
        double E = 0.5 * Math.pow(vC - y, 2);
        double dEdvC = vC - y;
        double dEduC = dEdvC * vC * (1 - vC);
        return new double[] {E, dEdvC, dEduC};
    }

    private static double[] weight(double[] x, double[] values, double y) {
        double[] outputLayerDerivatives = outputLayer(values, y);
        double[] hiddenLayerDerivatives = hiddenLayer(values, y);
        double[] dEdw = new double[array.length];
        dEdw[8] = values[3] * outputLayerDerivatives[2];
        dEdw[7] = values[1] * outputLayerDerivatives[2];
        dEdw[6] = outputLayerDerivatives[2];
        dEdw[5] = x[1] * hiddenLayerDerivatives[3];
        dEdw[4] = x[0] * hiddenLayerDerivatives[3];
        dEdw[3] = hiddenLayerDerivatives[3];
        dEdw[2] = x[1] * hiddenLayerDerivatives[1];
        dEdw[1] = x[0] * hiddenLayerDerivatives[1];
        dEdw[0] = hiddenLayerDerivatives[1];
        return dEdw;
    }

    private static double[] hiddenLayer(double[] values, double y) {
        double[] outputLayerDerivatives = outputLayer(values, y);
        double dEduC = outputLayerDerivatives[2];
        double dEdvA = array[7] * dEduC;
        double uA = values[0];
        double dEduA = uA >= 0 ? dEdvA : 0;
        double dEdvB = array[8] * dEduC;
        double uB = values[2];
        double dEduB = uB >= 0 ? dEdvB : 0;
        return new double[] {dEdvA, dEduA, dEdvB, dEduB};
    }

    private static double accuracy(ArrayList<Record> records) {
        int numAccuratePredictions = 0;
        for (Record record : records) {
            int prediction = eval(record.x)[5] >= 0.5 ? 1 : 0;
            int label = (int) record.y;
            if (prediction == label) {
                numAccuratePredictions++;
            }
        }
        return (double) numAccuratePredictions / records.size();
    }

    private static double setError(ArrayList<Record> records) {
        double setError = 0;
        for (Record record : records) {
            setError += Math.pow(eval(record.x)[5] - record.y, 2);
        }
        setError *= 0.5;
        return setError;
    }

    private static void stochasticGradientDescent(double[] weightDerivatives, double eta) {
        for (int i = 0; i < array.length; i++) {
            array[i] = array[i] - eta * weightDerivatives[i];
        }
    }

    private static void print(double[] values) {
        StringBuilder builder = new StringBuilder();
        for (double val : values) {
            builder.append(String.format("%.5f ", val));
        }
        System.out.println(builder.toString().trim());
    }

    public static void main(String[] args) {
        try {
            int flag = Integer.parseInt(args[0]);
            array = new double[9];
            for (int i = 0; i < array.length; i++) {
                array[i] = Double.parseDouble(args[i + 1]);
            }
            if (flag >= 100 && flag <= 500) {
                double y = Double.NaN;
                double eta = Double.NaN;
                double[] x = new double[2];
                for (int i = 0; i < x.length; i++) {
                    x[i] = Double.parseDouble(args[i + array.length + 1]);
                }
                if (flag >= 200) {
                    y = Double.parseDouble(args[array.length + x.length + 1]);
                }
                if (flag == 500) {
                    eta = Double.parseDouble(args[array.length + x.length + 2]);
                }

                if (flag == 100) {
                    print(eval(x));
                } else if (flag == 200) {
                    print(outputLayer(eval(x), y));
                } else if (flag == 300) {
                    print(hiddenLayer(eval(x), y));
                } else if (flag == 400) {
                    print(weight(x, eval(x), y));
                } else if (flag == 500) {
                    print(array);
                    print(new double[] {outputLayer(eval(x), y)[0]});
                    stochasticGradientDescent(weight(x, eval(x), y), eta);
                    print(array);
                    print(new double[] {outputLayer(eval(x), y)[0]});
                }
            } else {
                double eta = Double.parseDouble(args[array.length + 1]);
                ArrayList<Record> records = readFromFile(TRAIN);
                ArrayList<Record> evals = readFromFile(EVAL);
                ArrayList<Record> tests = readFromFile(TEST);
                int T = -1;
                if (flag >= 700) {
                    T = Integer.parseInt(args[array.length + 2]);
                }
                if (flag == 600) {
                    for (Record record : records) {
                        print(new double[] {record.x[0], record.x[1], record.y});
                        stochasticGradientDescent(weight(record.x, eval(record.x), record.y), eta);
                        print(array);
                        print(new double[] {setError(evals)});
                    }
                } else if (flag == 700) {
                    for (int i = 0; i < T; i++) {
                        for (Record record : records) {
                            stochasticGradientDescent(weight(record.x, eval(record.x), record.y), eta);
                        }
                        print(array);
                        print(new double[] {setError(evals)});
                    }
                } else if (flag == 800) {
                    int iter;
                    double prevSetError = Double.POSITIVE_INFINITY;
                    double setError = Double.NEGATIVE_INFINITY;
                    for (iter = 1; iter <= T; iter++) {
                        for (Record record : records) {
                            stochasticGradientDescent(weight(record.x, eval(record.x), record.y), eta);
                        }
                        setError = setError(evals);
                        if (setError > prevSetError) {
                            break;
                        } else {
                            prevSetError = setError;
                        }
                    }
                    System.out.println(iter == T + 1 ? iter - 1 : iter);
                    print(array);
                    print(new double[] {setError});
                    print(new double[] {accuracy(tests)});
                }
            }
        } catch (Exception e) {
            System.out.println("Usage: java Neural FLAG [args]");
        }
    }
}
