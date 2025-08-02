import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Eye, EyeOff, FileSpreadsheet, Zap } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";

const Login = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      // First test CORS connection
      console.log("Testing CORS connection...");
      const corsTest = await fetch("http://localhost:8000/test-cors", {
        method: "GET",
      });
      console.log("CORS test status:", corsTest.status);
      console.log("CORS test headers:", corsTest.headers);

      if (!corsTest.ok) {
        throw new Error("CORS test failed");
      }

      const formData = new FormData();
      formData.append("email", email);
      formData.append("password", password);

      console.log("Attempting login to:", "http://localhost:8000/auth/login");
      console.log("Email:", email);
      console.log("Password:", password);
      
      const response = await fetch("http://localhost:8000/auth/login", {
        method: "POST",
        body: formData,
        headers: {
          // Don't set Content-Type for FormData, let the browser set it
        },
      });

      console.log("Response status:", response.status);
      console.log("Response ok:", response.ok);
      console.log("Response headers:", response.headers);

      if (response.ok) {
        const data = await response.json();
        console.log("Login successful - Full response data:", data);
        
        // Store token and user info in localStorage
        localStorage.setItem("token", data.access_token);
        localStorage.setItem("user", JSON.stringify(data.user));
        
        console.log("Token stored:", data.access_token);
        console.log("User stored:", data.user);
        
        toast({
          title: "Login successful!",
          description: `Welcome back, ${data.user.full_name}!`,
        });

        console.log("Redirecting to upload page...");
        // Redirect to upload page instead of dashboard
        navigate("/upload");
      } else {
        const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
        console.error("Login failed:", errorData);
        toast({
          title: "Login failed",
          description: errorData.detail || "Invalid email or password",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Login error:", error);
      toast({
        title: "Login failed",
        description: "Network error. Please check if the backend server is running on http://localhost:8000",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary via-primary-variant to-secondary flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo/Brand Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 glass-morphism rounded-2xl mb-4">
            <div className="flex items-center space-x-1">
              <FileSpreadsheet className="h-6 w-6 text-white" />
              <Zap className="h-4 w-4 text-accent" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">AI OCR Invoice</h1>
          <p className="text-white/70">Welcome back! Please sign in to continue.</p>
        </div>

        {/* Login Form */}
        <Card className="glass-morphism shadow-elegant">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl text-foreground">Sign In</CardTitle>
            <CardDescription className="text-muted-foreground">
              Enter your credentials to access your dashboard
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="email" className="text-foreground font-medium">
                  Email Address
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="Enter your email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="bg-background/50 border-border text-foreground placeholder:text-muted-foreground focus:border-primary focus:ring-primary/20"
                  required
                  disabled={isLoading}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-foreground font-medium">
                  Password
                </Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="bg-background/50 border-border text-foreground placeholder:text-muted-foreground focus:border-primary focus:ring-primary/20 pr-10"
                    required
                    disabled={isLoading}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                    disabled={isLoading}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between text-sm">
                <label className="flex items-center space-x-2 text-muted-foreground">
                  <input
                    type="checkbox"
                    className="rounded border-border bg-background text-primary focus:ring-primary/20"
                    disabled={isLoading}
                  />
                  <span>Remember me</span>
                </label>
                <button
                  type="button"
                  className="text-primary hover:text-primary/80 transition-colors"
                  disabled={isLoading}
                >
                  Forgot password?
                </button>
              </div>

              <Button
                type="submit"
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 rounded-xl shadow-primary transition-all duration-300 transform hover:scale-[1.02]"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin w-5 h-5 border-2 border-current border-t-transparent rounded-full mr-2"></div>
                    Signing In...
                  </>
                ) : (
                  "Sign In"
                )}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-muted-foreground text-sm">
                Don't have an account?{" "}
                <button className="text-primary hover:text-primary/80 transition-colors font-medium">
                  Contact your administrator
                </button>
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="text-center mt-8 text-white/50 text-sm">
          <p>© 2024 AI OCR Invoice. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
};

export default Login;