  #define pump 22
#define convey 24

float x;
float y;
float z;
int en = 8;
int hom1 = 0;
int hom2 = 0;
int hom3 = 0;
int t1;
int t2;
int t3;
int dirX = 5;
int dirY = 6;
int dirZ = 7;
int stepX = 2;
int stepY = 3;
int stepZ = 4;
int endPinX = 9;
int endPinY = 10;
int endPinZ = 11;
int a = 0;
float current1 = 0;
float current2 = 0;
long steps_axis1 = 0;
long steps_axis2 = 0;
float ratio1 = 0;
float ratio2 = 0;
int pulseMax = 0;   
int pulseMin1 = 0; 
int pulseMin2 = 0; 
float maxstep;
float minstep1;
float minstep2;
int stepmax;
int stepmin1;
int stepmin2;

String data_ser1;
String data;
String vt[5];
int theta1, theta2, theta3;
String str_pump;
String str_convey;
int stt_pump = 1;
int stt_convey = 1;

//Khai bao ham
void kinematic (int theta1, int stepX, int theta2, int stepY, int theta3, int stepZ, int dela);

void tachViTri(String data,String *vt1,String *vt2,String *vt3,String *vt4,String *vt5){
  int a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0;
  for(int i=0; i<data.length(); i++) //tach gia tri goc cua chuoi nhan duoc tu python
    {
      if(data.charAt(i)==','){
        a1 = a2;
        a2 = a3;
        a3 = a4;
        a4 = a5; 
        a5 = i; 
      }
    }
    
    if(data.charAt(data.length() - 1) =='\n'){
      *vt1 = data;
      *vt2 = data;
      *vt3 = data;
      *vt3 = data;
      *vt4 = data;
      *vt5 = data;

      vt1->remove(a1 +1);
      vt2->remove(a2 + 1);
      vt2->remove(0,a1+1);
      vt3->remove(a3 + 1);
      vt3->remove(0,a2+1);
      vt4->remove(a4 + 1);
      vt4->remove(0,a3+1);
      vt5->remove(a5 + 1);
      vt5->remove(0,a4+1);
    }
}

void tachGoc(String data, int *theta1, int *theta2, int *theta3){
  int a1 = 0, a2 = 0;
  String chuoi1, chuoi2, chuoi3;
//  Serial1.print(data);
  for(int i=0; i<data.length(); i++) //tach gia tri goc cua chuoi nhan duoc tu python
    {
      if(data.charAt(i)==' '){
        a1 = a2;
        a2 = i; 
      }
    }
    
    if(data.charAt(data.length() - 1) ==','){
      chuoi1 = data;
      chuoi2 = data;
      chuoi3 = data;
      chuoi1.remove(a1);
      chuoi2.remove(a2);
      chuoi2.remove(0,a1+1);
      chuoi3.remove(data.length()-1);
      chuoi3.remove(0,a2+1);
    }

    *theta1 = chuoi1.toInt();
    *theta2 = chuoi2.toInt();
    *theta3 = chuoi3.toInt();
}

void tachGocKinematic(String data, int *theta1, int *theta2, int *theta3){
  int a1 = 0, a2 = 0;
  String chuoi1, chuoi2, chuoi3;
//  Serial1.print(data);
  for(int i=0; i<data.length(); i++) //tach gia tri goc cua chuoi nhan duoc tu python
    {
      if(data.charAt(i)==' '){
        a1 = a2;
        a2 = i; 
      }
    }
    
    if(data.charAt(data.length() - 2) =='1'){
      chuoi1 = data;
      chuoi2 = data;
      chuoi3 = data;
      chuoi1.remove(a1);
      chuoi2.remove(a2);
      chuoi2.remove(0,a1+1);
      chuoi3.remove(data.length()-2);
      chuoi3.remove(0,a2+1);
    }

    *theta1 = chuoi1.toInt();
    *theta2 = chuoi2.toInt();
    *theta3 = chuoi3.toInt();
}

void kinematic (int theta1, int stepX, int theta2, int stepY, int theta3, int stepZ, int dela)
{
  x = 0;
  y = 0;
  z = 0;
  int pulseMax = 0;   
  int pulseMin1 = 0; 
  int pulseMin2 = 0; 
  
  if (theta1 > t1)
  {
   digitalWrite(dirX,HIGH);
   x = 0.5555555555555555*(theta1-t1);
  }
  else if (theta1 < t1)
  { 
   digitalWrite(dirX,LOW);
   x = 0.5555555555555555*(t1-theta1);
  }
  t1 = theta1;
  
//----------- T2 ---------------//

  if (theta2 > t2)
  {
   digitalWrite(dirY,HIGH);
   y = 0.5555555555555555*(theta2-t2);
  }
  else if (theta2 < t2)
  { 
   digitalWrite(dirY,LOW);
   y = 0.5555555555555555*(t2-theta2);
  }
  t2 = theta2;

  //----------- T3 ---------------//

  if (theta3 > t3)
  {
   digitalWrite(dirZ,HIGH);
   z = 0.5555555555555555*(theta3-t3);
  }
  else if (theta3 < t3)
  { 
   digitalWrite(dirZ,LOW);
   z  = 0.5555555555555555*(t3-theta3);
  }
  t3 = theta3;
  
  maxstep = x;
  minstep1 = y;
  minstep2 = z;
  stepmax = stepX;
  stepmin1 = stepY;
  stepmin2 = stepZ;
   
  if(y > maxstep)
  {
    maxstep = y;
    minstep1 = x;
    minstep2 = z;
    stepmax = stepY;
    stepmin1 = stepX;
    stepmin2 = stepZ;
  }
  if(z > maxstep)
  {
    maxstep = z;
    minstep1 = x;
    minstep2 = y;
    stepmax = stepZ;
    stepmin1 = stepX;
    stepmin2 = stepY;
  }
  ratio1 = maxstep/minstep1;
  ratio2 = maxstep/minstep2;
  
  for(int i = 1 ; i <= (((maxstep/11)*57)*4) ; i=i+1) 
  {
    current1 = i / ratio1;
    current2 = i / ratio2;
    if (current1 - steps_axis1 >= 1)
    {
       digitalWrite(stepmin1,HIGH); 
       steps_axis1++;
    }
    if (current2 - steps_axis2 >= 1)
    {
       digitalWrite(stepmin2,HIGH); 
       steps_axis2++;
    }
    digitalWrite(stepmax,HIGH);
    delayMicroseconds(dela);
    digitalWrite(stepmax,LOW);
    digitalWrite(stepmin1,LOW); 
    digitalWrite(stepmin2,LOW); 
    delayMicroseconds(dela);
  }
  steps_axis1 = 0;
  steps_axis2 = 0;
}
void setup() 
{
  pinMode(pump, OUTPUT);
  pinMode(convey, OUTPUT);
  digitalWrite(pump, HIGH);
  digitalWrite(convey, HIGH);
  
  Serial.begin(115200);
  Serial.setTimeout(10);
  Serial1.begin(115200);
  Serial1.setTimeout(10);
  Serial2.begin(115200);
  Serial2.setTimeout(10);
  
  pinMode(en,OUTPUT);
  pinMode(dirX,OUTPUT);
  pinMode(stepX,OUTPUT);
  pinMode(dirY,OUTPUT);
  pinMode(stepY,OUTPUT);
  pinMode(dirZ,OUTPUT);
  pinMode(stepZ,OUTPUT);
  pinMode(endPinX,INPUT_PULLUP);
  pinMode(endPinY,INPUT_PULLUP);
  pinMode(endPinZ,INPUT_PULLUP);
  digitalWrite(en,LOW);
  digitalWrite(dirX,LOW);
  digitalWrite(dirY,LOW);
  digitalWrite(dirZ,LOW);
  t1 = 0;
  t2 = 0;
  t3 = 0;

  int t = 180 ;
  if(t != hom1)   
  {
    float c = 0.5555555555555555*t;
    for(int i = 1 ; i <= (((c/11)*57)*4) ; i=i+1)
    { 
        if (digitalRead(endPinX)== 1)  
        {
          digitalWrite(stepX,HIGH);
          delayMicroseconds(100); 
        }
        if (digitalRead(endPinY)== 1) 
        {
          digitalWrite(stepY,HIGH);
          delayMicroseconds(100); 
        }
        if (digitalRead(endPinZ)== 1) 
        {
          digitalWrite(stepZ,HIGH);
          delayMicroseconds(100); 
        }
        delayMicroseconds(700);
        digitalWrite(stepX,LOW);
        digitalWrite(stepY,LOW);
        digitalWrite(stepZ,LOW);
        delayMicroseconds(700);
    }
    hom1 = t; 
  }
}



void loop() 
{   
//  kinematic(vitridat1,stepX,vitridat2,stepY,vitridat3,stepZ,700);
  
  if(Serial.available() > 0)
  {
    data = Serial.readString();
    Serial2.println(data);

    if(data.charAt(data.length() - 2) =='2'){
      tachViTri(data, &vt[0], &vt[1], &vt[2], &vt[3], &vt[4]);
//    Serial2.println(vt[0]);
//    Serial2.println(vt[1]);
//    Serial2.println(vt[2]);
//    Serial2.println(vt[3]);
//    Serial2.println(vt[4]);

      for(int i = 0; i < 5; i++){
        if(i >=0 && i <4){
          digitalWrite(pump, LOW);
          if(i == 1){
            delay(1300);// set thoi gian doi vat
          }
        }
        else{
          digitalWrite(pump, HIGH);
          delay(300);
        }
        tachGoc(vt[i], &theta1, &theta2, &theta3);
        Serial2.print(theta1);
        Serial2.print(" ");
        Serial2.print(theta2);
        Serial2.print(" ");
        Serial2.println(theta3);
        kinematic(theta1,stepX,theta2,stepY,theta3,stepZ,800);
        delay(500);
      }
    }
    if(data.charAt(data.length() - 2) =='1'){
        tachGocKinematic(data, &theta1, &theta2, &theta3);
        Serial2.print(theta1);
        Serial2.print(" ");
        Serial2.print(theta2);
        Serial2.print(" ");
        Serial2.println(theta3);
        kinematic(theta1,stepX,theta2,stepY,theta3,stepZ,800);
      }

    if(data.charAt(data.length() - 2) =='3'){
      str_pump = data.charAt(0);
      stt_pump = str_pump.toInt();
      digitalWrite(pump, stt_pump);
    }

    if(data.charAt(data.length() - 2) =='4'){
      str_convey = data.charAt(0);
      stt_convey = str_convey.toInt();
      digitalWrite(convey, stt_convey);
    }
  }
//  if(Serial1.available() > 0){
//    data_ser1 = Serial1.readString();
//    Serial2.println(data_ser1);
//    if(data_ser1 == "a"){
//      digitalWrite(pump, LOW);
//    }
//    if(data_ser1 == "b"){
//      digitalWrite(pump, HIGH);
//    }
//    if(data_ser1 == "c"){
//      digitalWrite(convey, LOW);
//    }
//    if(data_ser1 == "d"){
//      digitalWrite(convey, HIGH);
//    }
//  }
}
