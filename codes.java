//caesar cipher
public class Caesercipherprac1 {
    public static void main(String[] args) {
        // TODO code application logic here
        String message, encryptedmessage="";
        int key;
        char ch;
        Scanner sc=new Scanner(System.in);
        System.out.println("Enter a message: ");
        message=sc.nextLine();

        System.out.println("Enter Key: ");
        key=sc.nextInt();
        for(int i=0;i<message.length();i++)
        {
            ch=message.charAt(i);

            if(ch>='a' && ch<='z')
            {
                ch=(char)(ch+key);
                if(ch>'z') {ch=(char)(ch-'z'+'a'-1);}
                encryptedmessage+=ch;

            }

            else if(ch>='A' && ch<='Z')
            {
                ch=(char)(ch+key);
                if(ch>'Z') {ch=(char)(ch-'Z'+'A'-1);}
                encryptedmessage+=ch;
            }

            else {encryptedmessage+=ch;}
        }

        System.out.println("Encrypted Message: "+encryptedmessage);
    }
}


// Monoalphabetic Cipher

package practical 1b;
import java.util.Scanner;
public class PractB {
    public static char p[] = { 'a', 'b', 'c', 'd', 'e','f','g','h','i','j','k','l','m','n'
        , 'o', 'p','q','r','s','t','u','v','w','x','y','z'
    };
    public static char ch[] = {
        'Q','W','E','R','T','Y','U','I','O','P','A','S','D','F','G','H','J','K','L','Z','X','C','V','B','N','M'
    };
    public static String doEncryption(String s){
        char c[] = new char[(s.length())];
        for (int i = 0; i < s.length(); i++){
            for (int j = 0; j < 26; j++){
                if (p[j] == s.charAt(i)){
                    c[i] = ch[j];
                    break;
                }}}
        return new String(c);
    }
    public static String doDecryption(String s){
        char p1[] = new char [(s.length())];
        for (int i = 0; i < s.length(); i ++){
            for (int j = 0; j < 26; j++){
                if(ch[j] == s.charAt(i)){
                p1[i] = p[j];
                break;
            }}}
        return new String(p1);
}
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter Message: ");
        String en = doEncryption(sc.next().toLowerCase());
        System.out.println("Encrypted Message: "+ en);
        System.out.println("Decrypted Message: "+ doDecryption(en));
        sc.close();
    }}

// Vernam Cipher

import java.util.*;

public class Prac2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Please enter a String: ");
        String str = sc.nextLine();
        String st = "";
        String otp;
        System.out.println("Enter OTP(One Time Pad): ");
        otp = sc.nextLine();
        char m, n;
        int p1 = 0, p2 =0;
        char c[] = new char[] { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };
        int n1[] = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                25 };
        if (str.length() != otp.length()) {

            System.out.println("Please enter OTP as the same length of string: ");
            otp = sc.nextLine();
        }
        for (int i = 0; i < str.length(); i++) 
        {
            m = (char) (str.charAt(i));
            n = (char) (otp.charAt(i));
            for (int j = 0; j<c.length; j++) 
            {
                if (m == c[j]) 
                {
                    p1 = n1[j];
                }
                if (n == c[j]) 
                {
                    p2 = n1[j];
                }
            }
            int p = p1 + p2;
            System.out.println(p1 + "+" + p2 + " = ");
            System.out.println(p);
            if (p > 26) {
                p = p - 26;
            }
            char c1 = c[p];
            System.out.println("\n\tCHARACTER at " + p + " is " + c1);
            st = st + c1;
        }
        System.out.println("---------------------------------------------");
        System.out.println("Cipher text is " + st);
    }
}

// Railfence cipher

import java.util.Scanner;
public class prac3 {
        public static void main(String[] args) {
        String rf;
        String s="";
        Scanner sc=new Scanner(System.in);
        System.out.println("Please enter a string: ");
        rf=sc.nextLine();
        int i;
        System.out.println("string="+rf);
        for(i=0;i<rf.length();i=i+2)
        {
            char c=rf.charAt(i);
            s=s+c;
            System.out.print(c);
        }
        for(int j=1;j<rf.length();j=j+2)
        {
            char c=rf.charAt(j);
            s=s+c;
            System.out.print(c);
        }
        System.out.println("\nCipher text is="+s);
        
    }
        
}


// Simple columnar 
import java.util.*;

public class Prac3b {
    public static void main(String[] args) {
        String text;
        int key1;
        int key[] = new int[4];
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter a message : ");
        text = sc.nextLine();
        char a[][] = new char[50][4];
        int l = text.length();
        int row;
        if (l % 4 == 0) {
            row = l / 4;
        } else {
            row = (l / 4) + 1;
        }
        int k = 0;
        System.out.println("\nMatrix: ");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < 4; j++) {
                a[i][j] = text.charAt(k);
                k++;
                System.out.print(a[i][j] + " ");
                if (l == k) {
                    break;
                }
            }
            System.out.println(" \n");
        }
        String s = "";
        System.out.println("Enter a key: ");
        for (int i = 0; i < 4; i++) {
            key[i] = sc.nextInt();
        }
        for (int i = 0; i < 4; i++) {

            key1 = key[i];
            for (int j = 0; j < row; j++) {
                String c = a[j][key1] + " ";
                if (c != "\0") {
                    s = s + c;
                }
            }
        }
        System.out.println("Cipher text: " + s);
    }
}

// DES
package pkg504_prac_des;
import java.util.*;
import javax.crypto.*;
public class Main {
    public static void main(String[] args) {
        try{
            Scanner sc = new Scanner(System.in);
            String s;
            System.out.println("Enter String: ");
            s= sc.nextLine();
            KeyGenerator key = KeyGenerator.getInstance("DES");
            SecretKey sk = key.generateKey();
            Cipher cip = Cipher.getInstance("DES");
            cip.init(Cipher.ENCRYPT_MODE, sk);
            byte[] encrypted = cip.doFinal(s.getBytes());
            cip.init(Cipher.DECRYPT_MODE, sk);
            byte[] decrypted = cip.doFinal(encrypted);
           System.out.println("Encrypted=" +new String(encrypted));
            System.out.println("Decrypted=" +new String(decrypted));
        }
catch(Exception e){
            System.out.println(e);
        }
    }
}


// AES
package pkg504_prac_des;
import java.util.*;
import javax.crypto.*;
public class Main {
    public static void main(String[] args) {
        try{
            Scanner sc = new Scanner(System.in);
            String s;
            System.out.println("Enter String: ");
            s= sc.nextLine();
            KeyGenerator key = KeyGenerator.getInstance("AES");
            SecretKey sk = key.generateKey();
            Cipher cip = Cipher.getInstance("AES");
            cip.init(Cipher.ENCRYPT_MODE, sk);
            byte[] encrypted = cip.doFinal(s.getBytes());
            cip.init(Cipher.DECRYPT_MODE, sk);
            byte[] decrypted = cip.doFinal(encrypted);
            System.out.println("Encrypted=" +new String(encrypted));
            System.out.println("Decrypted=" +new String(decrypted));
        }catch(Exception e){
            System.out.println(e);
        }
    }
}


// RSA
package ins_prac5;
import java.math.BigInteger;
import java.util.Random;
import java.util.Scanner;

public class INS_prac5 {
  
    public static void main(String[] args) {
      Scanner sc = new Scanner(System.in);
      System.out.println("Enter plain text: ");
      BigInteger pt = new BigInteger(sc.next());
      System.out.println("Enter 2 Prime number:");
      
      BigInteger p = new BigInteger(sc.next());
      BigInteger q = new BigInteger(sc.next());
      BigInteger n = p.multiply(q);
      
      BigInteger one = new BigInteger("1");
      BigInteger two = p.subtract(one);
      BigInteger three = q.subtract(one);
      BigInteger four = two.multiply(three);
      
      BigInteger e;
      do{
          e = new BigInteger(2*512, new Random());
      }
      while((e.compareTo(four) != 1) || (e.gcd(four).compareTo(one)) !=0);
      System.out.println("Public Key is: " + e);
      BigInteger d = e.modInverse(four);
      System.out.println("private key is: "+ d);
      BigInteger ct = pt.modPow(e, n);
      
      System.out.println("Cipher Text is: "+ ct);
      BigInteger ptl = ct.modPow(d,n);
      System.out.println("Plain txt is: "+ ptl);
    }
    
}


// Diffie Hellman
import java.util.*;
public class INS_prac6 {
public static void main(String[] args) {
Scanner sc = new Scanner(System.in);
System.out.println("Enter module(p)");
int p = sc.nextInt();
System.out.println("Enter primitive root of "+p);
int g = sc.nextInt();
System.out.println("Choose 1st Secret Key (ALICE): ");
int a = sc.nextInt();
System.out.println("Choose 2nd Secret Key (BOB): ");
int b = sc.nextInt();
int A = (int)Math.pow(g,a)%p;
int B = (int)Math.pow(g,b)%p;
int S_A = (int)Math.pow(B,a)%p;
int S_B = (int)Math.pow(A,b)%p;
if(S_A == S_B){
System.out.println("Alice and Bob can communicate");
}else{
System.out.println("Error");
}
}
}


// MD-5
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
public class INS_prac7 {
public static void main(String[] args) {
System.out.println("Hashcode by MD5 for: ");
System.out.println("\n"+"For String"+"\n"+ encrypt("Hello World"));
System.out.println("\n"+"For Numbers"+"\n"+ encrypt("12345"));
System.out.println("\n"+"For Null"+"\n"+ encrypt(null));
}
public static String encrypt(String input){
if(input == null){
return null;
}
try{
MessageDigest md = MessageDigest.getInstance("MD5");
byte[] MessageDigest = md.digest(input.getBytes());
BigInteger no = new BigInteger(1, MessageDigest);
String hashtext = no.toString(16);
return hashtext;
}catch(NoSuchAlgorithmException e){
throw new RuntimeException(e);
}
}
}

// HMAC-SHA1

import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
public class INS_Prac8 {
public static void main(String[] args) {
System.out.println("HashCode by SHA-1 for");
System.out.println("\n"+ "Hello World"+"\n"+ encrypt("Hello World"));
}
public static String encrypt(String input){
try{
MessageDigest md = MessageDigest.getInstance("SHA-1");
byte[] MessageDigest = md.digest(input.getBytes());
BigInteger no = new BigInteger(1, MessageDigest);
String hashtext = no.toString(16);
return hashtext;
}catch(NoSuchAlgorithmException e){
throw new RuntimeException(e);
}
}
}