package com.amicabile.spamclassifier;

public class Record {
    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public Record(String message, int label) {
        this.message = message;
        this.label = label;
    }

    private String message;
    private int label;

}
