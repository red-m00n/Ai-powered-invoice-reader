import { Navigate } from "react-router-dom";

interface AdminRouteProps {
  children: React.ReactNode;
}

const AdminRoute = ({ children }: AdminRouteProps) => {
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const isAdmin = user.role === "admin";

  if (!isAdmin) {
    return <Navigate to="/upload" replace />;
  }

  return <>{children}</>;
};

export default AdminRoute; 