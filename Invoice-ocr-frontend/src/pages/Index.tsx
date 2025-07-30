import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Upload, FileText, CheckCircle } from "lucide-react";
import Navbar from "@/components/Navbar";

const Index = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setIsComplete(false);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      toast({
        title: "No file selected",
        description: "Please select an invoice file to upload.",
        variant: "destructive",
      });
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });
      setUploadProgress(100);
      if (response.ok) {
        setIsComplete(true);
        toast({
          title: "Success!",
          description: "Invoice processed and saved successfully.",
        });
      } else {
        const data = await response.json();
        toast({
          title: "Upload failed",
          description: data.error || "An error occurred during upload.",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Upload failed",
        description: (error as Error).message,
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setUploadProgress(0);
    setIsComplete(false);
    setIsUploading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background-secondary to-background-tertiary">
      <Navbar />
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-16 animate-fade-in">
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-6">
              <span className="text-sm font-medium text-primary">✨ AI-Powered Technology</span>
            </div>
            <h1 className="text-6xl font-display font-bold mb-6 bg-gradient-to-r from-primary via-primary-variant to-secondary bg-clip-text text-transparent">
              OCR for Invoices
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Transform your invoice processing with cutting-edge AI technology. 
              Upload, extract, and manage invoice data in seconds.
            </p>
          </div>

          {/* Upload Card */}
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-3xl blur-xl transform scale-110"></div>
            <Card className="relative gradient-card shadow-elegant border-0 backdrop-blur-sm hover-lift">
              <CardHeader className="pb-8">
                <CardTitle className="flex items-center gap-3 text-2xl font-display">
                  <div className="p-3 rounded-xl gradient-primary">
                    <FileText className="w-6 h-6 text-white" />
                  </div>
                  Upload Your Invoice
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-8">
                {!isUploading && !isComplete && (
                  <div className="animate-slide-up">
                    <div className="space-y-6">
                      <div className="space-y-3">
                        <Label htmlFor="file" className="text-base font-medium">Select Invoice File</Label>
                        <div className="relative">
                          <Input
                            id="file"
                            type="file"
                            accept=".pdf,.jpg,.jpeg,.png"
                            onChange={handleFileChange}
                            className="cursor-pointer h-14 text-base border-2 border-dashed border-primary/30 bg-primary/5 hover:border-primary/50 transition-all duration-300"
                          />
                          <Upload className="absolute right-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                        </div>
                        <p className="text-sm text-muted-foreground flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-success"></span>
                          Supported formats: PDF, JPG, PNG (Max 10MB)
                        </p>
                      </div>

                      {file && (
                        <div className="p-6 border-2 border-primary/20 rounded-xl bg-gradient-to-r from-primary/5 to-secondary/5 animate-scale-in">
                          <div className="flex items-center gap-4">
                            <div className="p-3 rounded-lg gradient-secondary">
                              <FileText className="w-5 h-5 text-white" />
                            </div>
                            <div className="flex-1">
                              <p className="font-semibold text-foreground">{file.name}</p>
                              <p className="text-sm text-muted-foreground">
                                {(file.size / 1024 / 1024).toFixed(2)} MB • Ready to process
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      <Button 
                        onClick={handleUpload} 
                        disabled={!file} 
                        className="w-full h-14 text-lg font-semibold gradient-primary hover:shadow-glow transition-all duration-300 disabled:opacity-50"
                        size="lg"
                      >
                        <Upload className="w-5 h-5 mr-3" />
                        Process with AI
                      </Button>
                    </div>
                  </div>
                )}

                {isUploading && (
                  <div className="space-y-8 text-center animate-fade-in">
                    <div className="relative">
                      <div className="w-20 h-20 border-4 border-primary/30 border-t-primary rounded-full animate-spin mx-auto mb-6"></div>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-12 h-12 bg-gradient-primary rounded-full animate-glow"></div>
                      </div>
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">AI is Processing Your Invoice</h3>
                      <p className="text-muted-foreground">Advanced OCR technology is extracting data...</p>
                    </div>
                    <div className="space-y-4">
                      <Progress value={uploadProgress} className="w-full h-3" />
                      <div className="flex justify-between text-sm font-medium">
                        <span className="text-primary">{uploadProgress}% Complete</span>
                        <span className="text-muted-foreground">Processing...</span>
                      </div>
                    </div>
                  </div>
                )}

                {isComplete && (
                  <div className="text-center space-y-6 animate-scale-in">
                    <div className="relative">
                      <div className="w-20 h-20 bg-gradient-to-r from-success to-green-400 rounded-full mx-auto flex items-center justify-center animate-glow">
                        <CheckCircle className="w-10 h-10 text-white" />
                      </div>
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-success mb-3">Processing Complete!</h3>
                      <p className="text-muted-foreground text-lg">
                        Your invoice has been successfully processed and all data has been extracted and saved.
                      </p>
                    </div>
                    <div className="flex gap-4 justify-center">
                      <Button onClick={resetForm} variant="outline" className="font-semibold">
                        Upload Another Invoice
                      </Button>
                      <Button className="gradient-primary font-semibold">
                        View in Dashboard
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 mt-16 animate-slide-up">
            {[
              {
                icon: "🚀",
                title: "Lightning Fast",
                description: "Process invoices in seconds with our advanced AI"
              },
              {
                icon: "🎯",
                title: "99% Accuracy",
                description: "Industry-leading precision in data extraction"
              },
              {
                icon: "🔒",
                title: "Secure & Private",
                description: "Your data is encrypted and protected"
              }
            ].map((feature, index) => (
              <Card key={index} className="text-center p-6 hover-lift border-0 gradient-card">
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="font-display font-semibold text-lg mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
